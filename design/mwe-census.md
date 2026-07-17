# The MWE census — is the E2a corpus reconstructible at all? Registered before reading

**Status:** REGISTRATION, 2026-07-18. Committed before any of the 13
threads is re-read for this question and before any E2a harness beyond
the positive control is written. This census bounds what the E2a band can
ever say: *≥4 of 13* reads one way if thirteen are reconstructible and
another if six are — a 4/13 with seven non-reconstructible isn't a 4/13.
The coverage lesson, third venue, settled before the run instead of
renegotiated after it.

## The corpus, pinned — and a membership correction, recorded loudly

The E2a corpus is **the 13 state-predicate hits** of
`design/layer-probe.md` §3, verbatim:

> dfx#207, dfx#223, dfx#417, dfx#632, dfx#657, dfx#752, jmd#339,
> bjx#D416, bjx#969, bjx#973, npy#249, npy#552, npy#1133

**Correction: hit386 is not one of them.** dfx#386's registered property
is termination — the layer probe's *trajectory* bucket — and the value
model names its box as the **positive control** ("beginning with
re-deriving hit386's own box"), not as a corpus member. Prior status
lines reporting "1 of 13, on the positive control" conflated the control
with the corpus; the correct statement is: **0 of 13 mechanized; positive
control passed** (VERIFIED, re-certified under (i) and (ii)). "The
twelve" in recent work orders was arithmetic on the same error: the
remaining cases number thirteen. The falsifier clause "0 including
failure to re-derive the hand-proved box" is unaffected — it always
referred to the control, and the control passed.

## Buckets — fixed before reading

| bucket | meaning |
|---|---|
| **full MWE** | a runnable system with concrete constants **in the thread** (body or comments) — the field/model can be transcribed without inventing a value |
| **partial** | the system named or code present but constants/pieces missing, or prose only |
| **none** | no reproducible system |

Operational rules, fixed now:

- **Reconstructible = full MWE only.** Partial and none are excluded from
  E2a's denominator **before the run**, and the exclusion list is
  published with the per-hit reason.
- **Linked-external code does not count** — "in the thread" means
  inlined. A link can rot, and a link's target can change after the
  incident; the census reads what the thread fixed.
- Why this can't be caught downstream: for an invented system, (i) is
  circular by construction — invent the system, invent `y0`, state a box
  containing the invented `y0`, and nonvacuity checks green. Perfectly
  vacuous, perfectly certified. The census is the only place this is
  visible.
- The direction is conservative: excluding **raises** the bar (≥4 of R
  for R < 13) without moving the threshold. That is the safe way to be
  wrong.

## The arithmetic contingency — fixed now, before anyone writes a harness

Let **R** = the full-MWE count. **If R < 4, the supported band is
unreachable as registered, and that is not a run and not a
falsification** — the model would be dead by arithmetic, not by
evidence. In that case: the E2a run is **suspended before any case
runs**, the unreachability is reported as a denominator fact, and E2a's
bands are re-registered against R — openly, with the original bands'
text retained and the re-registration dated. Falsification-by-arithmetic
is not a permitted reading, in either direction.

## Prior, recorded

From the attribution probe's full-thread reads (all 13 threads were read
this week for a different variable): several bodies visibly carry
complete scripts (npy#249, dfx#632, dfx#657 at minimum), several are
visibly prose-or-fragment (bjx#973 is a design question; dfx#752 is a
mechanism report). Expected R: **7–10 of 13**. If R lands below 4 the
contingency above fires; the proposer notes he does not expect it to.

---

# Reading (2026-07-18 — all 13 threads re-fetched with full bodies)

Code blocks extracted from every body and comment
(`$S/mwe/*.json`), then read by hand against the buckets.

| hit | bucket | grounds |
|---|---|---|
| **bjx#D416** | **full MWE** | body: complete predator-prey ODE + sampling script, concrete priors (`LogNormal(log(10),1)`, `TruncatedNormal(1.0,0.5,0,∞)`, …); depends on tfp + `odeint` but is transcribable without inventing a value |
| **bjx#969** | **full MWE** | body: runnable script, `logdensity_fn` with `bound=5.0`, `dim=2`, `initial_step_size=100.0` |
| **jmd#339** | **full MWE** | body: `cell0=100`, concrete `pos` array, cutoffs 5.0/5.5; runnable neighbor-list repro |
| **dfx#207** | **full MWE** | body: `funclog2` field + stoichiometry matrix inline; rate constants supplied **in-thread in a comment** (`k1=3.24e-4`, …) — provenance noted, still in-thread |
| **dfx#417** | **full MWE** | body: sigmoid/potential SDE system, `SEED=123`, concrete (block 1 is a conda dump, ignored) |
| **dfx#632** | **full MWE** | body: `-y/args[0]`, `tau=1e-11`, `dt0=tau/1000`, `y0=1.0` — minimal and complete |
| **dfx#657** | **full MWE** | body: StepTo MWE, `t0=0`, `t1=1e-9`, `dt=2e-12`, `f=-y/t1`, `y0=ones` (the `python 3`-fenced block is complete) |
| **npy#249** | **full MWE** | body: `dual_moon_pe` potential, `init_params=np.array([2.,0.])` |
| **npy#1133** | **full MWE** | body: model + run + concrete data (`np.random.normal(loc=1, scale=5, size=(500,7))`) |
| **bjx#973** | partial → **excluded** | prose design question; the system is #969's, *referenced* not inlined; 0 code blocks |
| **dfx#223** | partial → **excluded** | "8 coupled ODEs … time-varying interpolated inputs" described in prose; no constants inlined; the only in-thread code is diffrax-internal patch snippets, not the user's system |
| **dfx#752** | partial → **excluded** | mechanism report + a one-line diffrax-internal fix (`y_error = jtu.tree_map(...)`); the user's actual stiff ODE is never given |
| **npy#552** | partial → **excluded** | BNN model *function* present, but `X`, `Y`, and the weight inits are unfilled arguments — not runnable without inventing the data |

## R = 9. The band is reachable; the contingency does not fire.

**Reconstructible (full MWE): 9 of 13** — bjx#D416, bjx#969, jmd#339,
dfx#207, dfx#417, dfx#632, dfx#657, npy#249, npy#1133. Within the prior's
7–10, and ≥ 4, so the R<4 arithmetic contingency stays dormant and the
supported band (≥ 4 across ≥ 2 libraries) is reachable as registered.

**Excluded from E2a's denominator, before the run, with reasons above:**
bjx#973, dfx#223, dfx#752, npy#552. **E2a's denominator is now 9, not
13.** Per the conservative direction registered above, this *raises* the
bar — ≥ 4 of 9 is harder than ≥ 4 of 13 — without moving the threshold.
The ≥ 2-libraries clause still binds; the 9 span diffrax (5), blackjax
(2), jax-md (1), numpyro (2) — three libraries even after the two-way
blackjax and single jax-md entries, so a distributed result remains
possible.

## Recorded caveats

- **dfx#207's rate constants are maintainer-supplied** (in a comment, not
  the reporter's body). In-thread and pinned to the incident, so it meets
  the bucket; the provenance is logged for transparency.
- Reconstructibility is not tractability. Three full-MWE hits
  (bjx#D416, npy#1133, npy#552-were-it-in) are full sampling models whose
  E2a treatment may hit coverage or harness limits — but those are
  *later* buckets (`design/unknown-triage.md`), not reasons to exclude
  here. The census asks only: transcribe faithfully, or invent? For the
  9, transcribe.
- "0 of 13 mechanized, positive control passed" becomes **"0 of 9
  mechanized"** under the corrected denominator; the control (hit386) was
  never in either count.
