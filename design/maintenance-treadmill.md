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
  `num_consts`/`num_carry`/`linear`. Any consumer relying on scan's carry
  layout breaks; our interrogation classifier now reports
  `layout-unknown` on 0.11 rather than guessing.
- `Jaxpr` and `ClosedJaxpr` **merged into one class**: isinstance
  distinctions across that boundary silently changed meaning (caught by a
  test that pinned 0.10 behaviour; made series-tolerant).
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

**Cost of this bump:** one 4-line transcription rule (FTTuple → structural
text), two test adjustments, one interrogation guard; under an hour
wall-clock including re-verification of both series. `TESTED_JAX_SERIES`
is now `("0.10", "0.11")`.

## The rates — two, not one

**Name-level churn** — additive; one registry entry each; genuinely cheap.
Scales with the registry:

| bump | new prims | removed | lowering shifts |
|---|---|---|---|
| 0.10 → 0.11 | 1 (`unstack`) | 0 | `slice`+`squeeze` → `unstack` |

**Structural churn** — costs no registry entries and **invalidates
analyses**. Scales with the number of analyses stelling owns, so the
denominator is part of the measurement:

| bump | changes | analyses broken / owned |
|---|---|---|
| 0.10 → 0.11 | scan layout params removed; `Jaxpr`/`ClosedJaxpr` merged | **1 / ~2** — the counter-vs-carry classifier now reports `layout-unknown` on 0.11; the census/coverage counting survived |

The denominator is the honest part: "~4 lines, under an hour" was measured
at the single moment in this project's life when there is almost nothing
to break — and one of roughly two owned analyses broke anyway. Merged into
one number, this slope would lie at Stage 2, when twenty analyses exist
and the same structural bump costs a week. Track the ratio, not the LOC.

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
