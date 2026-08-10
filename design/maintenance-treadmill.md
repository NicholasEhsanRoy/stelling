# Maintenance treadmill — how fast the ground moves under a jaxpr-level tool

**Status:** evidence artifact, first measurement 2026-07-17. Method: on
each new jax series, re-run the census (`corpus/run_census.py`) and the
full suite, and diff the primitive profile and transcription behaviour
against the previous series. Append one row per bump — a rate needs two
points, and every future bump extends this for free. Nobody has published
this number for JAX; the value model currently prices this cost at zero.

## Bump 1: jax 0.10.2 → 0.11.0 (measured 2026-07-17, within a day of 0.11.0 landing)

Same corpus, same harnesses; both runs complete (8/8 public harnesses and
the held-out arm, on both series).

**Primitive-name level** (what the census table sees):

- appeared: `unstack` ×1 (blackjax HMC path)
- disappeared: none
- counts moved: `slice` 48→46, `squeeze` 12→10 — the `slice`+`squeeze`
  pattern became `unstack`: a *lowering* change, not a code change
  (blackjax 97→94 eqns)
- totals: 1823 → 1820 equations; 73 → 74 primitives

**Param level** (invisible to the count table; where the maintenance
actually lives):

- `scan` gained `ft_in`/`ft_out` — a new `flattree.FTTuple` structural
  type (the flat-tree redesign arriving in jax) — and **lost**
  `num_consts`/`num_carry`. Any consumer relying on scan's carry
  layout breaks; our interrogation classifier now reports
  `layout-unknown` on 0.11 rather than guessing. (An earlier version of
  this line also listed `linear` as lost in this bump. It was not: driven
  over six scan forms — plain, with consts, `length=`-only, tuple carry,
  under `grad`, `fori_loop` — `linear` is absent on **0.10.2 as well**,
  and `linear=` appears at no `scan_p.bind` site in 0.10.2's own
  `jax._src.lax.control_flow.loops`. It was lost in some earlier series;
  this bump did not lose it.)
- `Jaxpr` and `ClosedJaxpr` **merged into one class**: isinstance
  distinctions across that boundary silently changed meaning. **This was
  recorded here as "made series-tolerant" and it was not.** What the bump
  actually did was make ONE TEST tolerant and leave its consumers behind:
  `propagate._is_add_combiner` and two `remat2` body readers
  (`propagate`'s transparent descent and `obligation`'s slice validator)
  went on testing the *closed* shape only. On 0.10 that reads as "no
  combiner" and "no body", so **every `.at[].add` row declined on 0.10**
  — VERIFIED silently becoming UNKNOWN — and every remat'd wrapper was
  left opaque. Nothing caught it for three weeks because there was no
  0.10 lane to catch it with (see below). The repairs are `76140c2`
  (structural combiner read) and `9735576` (one canonical accessor,
  `coverage.call_body`), both **after** this bump, both prompted by the
  lane and not by the bump.
- `convert_element_type.new_dtype` is now a `numpy.dtypes.*` instance
  (benign — still an `np.dtype` subclass).

**Fence performance — second and third live tests.** The unknown-param
raise fired on `ft_in` exactly as designed: loud, naming the primitive and
the param. (Its first live test was the accidental 0.5.3 exposure, which
surfaced the old `custom_vjp_call_jaxpr` primitive name the same way —
renames across series gaps are real.) And the bump found a hole in a fence
itself: the untested-series warning is once-per-version, so the
series-friction test passed whenever any earlier test had already consumed
the warning — a test-order-dependent fence, fixed with a cache clear.
Fences need fences.

**Ecosystem lag: zero.** All seven corpus libraries installed and traced
on 0.11.0 on day one.

**Cost of this bump, as first recorded:** one 4-line transcription rule
(FTTuple → structural text), two test adjustments, one interrogation
guard; under an hour wall-clock "including re-verification of both
series".

**Cost of this bump, corrected 2026-08-07 — the re-verification of the
older series had not happened.** Every jax CI job installed
`.[solvers,jax]`, whose floor was `jax>=0.5` at the time (it is `jax>=0.10`
since `c3ff79d`) and which has never carried an **upper** bound — the half
that actually does the work here — so every job resolved the
NEWEST jax: whatever `TESTED_JAX_SERIES` said, CI exercised **one**
series, and 0.10 rested on a developer's run. When 0.10 finally got a
lane of its own (`1053714`), the suite on a real 0.10.2 failed **ten
tests** — nine of them the one `_is_add_combiner` container bug above,
the tenth `jit`'s `inline` param. So the honest cost is the hour, **plus**
three consumer repairs and a CI lane, found three weeks later. The
generalisable part is not the hour: it is that *the bump's own cost cannot
be measured on the series the bump moves to*. `TESTED_JAX_SERIES` is now
`("0.10", "0.11")`, and each entry now has a lane — an entry with no lane
is a claim, not a test.

## The rates — two, not one

**Name-level churn** — additive; one registry entry each; genuinely cheap.
Scales with the registry:

| bump | new prims | removed | lowering shifts |
|---|---|---|---|
| 0.10 → 0.11 | 1 (`unstack`) | 0 | `slice`+`squeeze` → `unstack` |

"New" here means **new to the census**, not new to jax: measured,
`jnp.unstack` traces to an `unstack` primitive on 0.10.2 as well. What
moved is blackjax's lowering, which is why the same row is filed as a
lowering shift. A registry gains an entry either way, so the cost is the
same — but the distinction matters for anyone reading this column as a
jax changelog.

**Structural churn** — costs no registry entries and **invalidates
analyses**. Scales with the number of analyses stelling owns, so the
denominator is part of the measurement:

| bump | changes | analyses broken / owned |
|---|---|---|
| 0.10 → 0.11 | scan layout params removed; `Jaxpr`/`ClosedJaxpr` merged | **2 / ~2** (was recorded 1 / ~2) — the counter-vs-carry classifier reports `layout-unknown` on 0.11; and the `Jaxpr`/`ClosedJaxpr` merge broke the transfer-and-emission path on the OLD series in three places. The census/coverage counting survived |

**Why the recorded figure was 1 and the measured figure is 2.** A
structural change breaks an analysis on the series you are *not* running.
At the time of the bump only 0.11 was running, so the merge looked
absorbed; it was not, and the 0.10 lane later billed for it. The
denominator is the honest part: "~4 lines, under an hour" was measured at
the single moment in this project's life when there is almost nothing to
break — and **both** of roughly two owned analyses broke, one of them
invisibly. Merged into one number, this slope would lie at Stage 2, when
twenty analyses exist and the same structural bump costs a week. Track the
ratio, not the LOC — and do not read a ratio taken on one lane.

**There is a third speed, and no instrument here reaches it (recorded
2026-07-18).** The census measures *structural* churn — primitives
appearing, params changing shape. A **semantics** change produces the
same jaxpr with a different meaning: `sqrt` is still `sqrt`, the count
does not move, the census sees nothing — and the archaeology probe's
headline receipt (`safe_mask`: "Fixed NaNs due to a JAX change in the
behavior of np.sqrt at 0") is a dated instance of exactly this class
invalidating downstream correctness silently. Whether the class has real
volume is what `design/semantics-drift-probe.md` measures; that this file
was blind to it is true today and is recorded today. If that probe
returns supported, the treadmill's cost acquires a second face: the same
versioned-transfer discipline it charges for is the only instrument that
can *detect* semantic drift downstream (in differential form — the
circularity limit in the probe's registration).

**Ecosystem lag of zero is the best news in this file** — seven mature
libraries traced on a day-old jax release. It is the only quantitative
evidence anyone has bearing on the no-upper-caps rule, and it points the
right way.
