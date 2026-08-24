# Maintenance treadmill — how fast the ground moves under a jaxpr-level tool

**Status:** evidence artifact, first measurement 2026-07-17. Method: on
each new jax **release** — not each series — re-run the census
(`corpus/run_census.py`) and the full suite, and diff the primitive profile
and transcription behaviour against the previous release. Append one row
per bump — a rate needs two points, and every future bump extends this for
free. Nobody has published this number for JAX; the value model currently
prices this cost at zero.

**THE UNIT WAS "SERIES" AND BUMP 2 FALSIFIED IT.** This line read *"on
each new jax series"*, and `ir.py`'s `_REQUIRED_PARAMS` said the same thing
in the same words. jax 0.11.0 → 0.11.1 is one series and moved both a
primitive's param set and the source of the const-fold rule the overflow
tripwire attaches to. A series is not the interval over which jax's jaxpr
surface holds still, and treating it as one means the instrument samples
at a rate the thing it measures exceeds.

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

## Bump 2: jax 0.11.0 → 0.11.1 (measured 2026-08-18)

**SCOPE, FIRST, because this row is narrower than Bump 1's and saying so is
the point of the file.** What was re-driven on both releases, in isolated
venvs on one machine (CPython 3.12.3, jax + jaxlib 0.11.0 and 0.11.1, CPU):
a 70-form param census (`param_key_census.py` in the sweeps repository,
extended), the full pytest suite in both `JAX_ENABLE_X64` cells, and the
const-fold rule the overflow tripwire attaches to. What was **not**
re-driven is `corpus/run_census.py` itself, for the reason in the ecosystem
row below — so the primitive **counts** of Bump 1 have no counterpart here,
and the name-level row below is scoped to the 70-form census rather than to
the corpus.

**Primitive-name level:** nothing. 70 forms reached 60 distinct primitives
on each release and the two sets are identical.

**Param level** (where this bump's whole cost is):

- `reduce_max` and `reduce_min` gained `out_sharding`. Exactly these two of
  the 60; `reduce_sum` already carried it on both. That is the counterexample
  that moved this file's unit from series to release.
- **No verdict moves.** Both primitives are absent from `TRANSFERS` and
  from `IEEE_TRANSFERS` on both releases, so no transfer reads the param.
  Driven end to end, `JAX_ENABLE_X64=1`: `assert_(jnp.max(a) > 0.0)` over
  `any_array((4,), float64, (0.1, 10.0))` returns UNKNOWN on both, with the
  same note — `1 equation(s) fell to ⊤ (reduce_max ×1)`. The cell is named
  because the NOTE is a fact about the cell and not only about the release:
  with `JAX_ENABLE_X64` unset the declared `float64` is truncated to
  `float32`, a `convert_element_type` enters the jaxpr, its transfer
  declines, and the note reads `2 equation(s) fell to ⊤
  (convert_element_type ×1, reduce_max ×1)` — on 0.11.0 and 0.11.1 alike.
  The STATUS is UNKNOWN in all four cells. `SOUNDNESS.md`'s entry carried
  this sentence without the qualifier and as a claim about both cells,
  which is the half that was wrong; it is corrected there.
- **Query identity moves, and silently.** `content_hash` hashes the params,
  so the same harness containing a max/min reduction traces to a different
  hash on the two releases. A stored document keeps its hash and still
  loads; a *fresh trace* compared against a stored hash mismatches with
  nothing raising. Recorded in `SOUNDNESS.md` under 2026-08-18; the
  measured pairs are here rather than there because
  `test_the_record_does_not_pin_a_hash_LITERAL_in_prose` forbids a hash
  literal in the record files, and this is an evidence artifact. Each row
  is `assert_(f(a) > 0.0)` over `any_array((4,), float64, (0.1, 10.0))`,
  `JAX_ENABLE_X64=1`, first 16 hex of `ClosedJaxpr.content_hash()`:

  | `f` | primitive | jax 0.11.0 | jax 0.11.1 |
  |---|---|---|---|
  | `jnp.max`, `.max()`, `jnp.amax` | `reduce_max` | 92a92863ae579bbd | ff25a9762c102215 |
  | `jnp.min`, `.min()`, `jnp.amin` | `reduce_min` | ac02bcadefbf6933 | 3eee32c7d405432d |
  | `jnp.maximum(a, 0.0)` | `max` | c295963f58b0aabd | *unchanged* |
  | `jnp.minimum(a, 100.0)` | `min` | 8576786984428216 | *unchanged* |
  | `jnp.sum` | `reduce_sum` | 52336382a4d6677b | *unchanged* |

  `jnp.max(a, axis=0)` over a `(2, 3)` declaration moves too
  (22092a65fdf3e6ba → adf3c4f9b61a5e6a), and `docs/quickstart.md`'s harness
  — which contains no reduction — does not (628a25efd4417f44 on both).

**Tripwire level** (invisible to both of the above): the const-fold rule
`_convert_elt_type_folding_rule` changed source, sha1 `c808b3001114` →
`522706b62a10`, by one line — `not np.shape(c)` → `not out_aval.shape`
(jax `803de7b08`, 2026-08-11). Semantics-preserving for this tool: the two
spellings differ only where `np.shape(t.get_const()) != t.aval.shape`,
which cannot arise over a `DynamicJaxprTrace` — proved there, and measured
in the qualification that preceded this row at 122,672 const-fold
invocations per release with zero disagreements (that figure is carried
forward here, not re-derived). Re-derived for this row: both hashes, the
one-line diff, and that the rule's NAME did not move. The shipped `arm()`
still arms on both releases and its live control still fires on both —
driven through `.github/scripts/tripwire_canary.py --require`, exit 0 on
each. What the bump cost was the version→hash map that replaced
`_KNOWN_HASH`: a single constant could not express *which* release carries
which rule, and 0.11.0 and 0.11.1 are one series carrying two.

**Fence performance.** One fence fired and it was the right one: running
the whole suite in one isolated venv per release, in both `JAX_ENABLE_X64`
cells, the rule-hash pin in `tests/test_tripwire_arm.py` was the **only**
test whose status differed between 0.11.0 and 0.11.1. Nothing else moved.
The fence that did **not** fire is `_REQUIRED_PARAMS`' — correctly:
`reduce_max`/`reduce_min` have no row there, and giving one to `reduce_max`
was measured to refuse an honest `jnp.max` document traced on 0.11.0.

**Ecosystem lag: NOT zero, for the first time — and it is not jax's.**
`flax` 0.12.8 cannot import `flax.nnx` on jax 0.11.1: jax removed
`jax.experimental.hijax.MutableHiType` and `AvalMutableQDD`, and
`flax/nnx/variablelib.py` subclasses the first at module scope. `jax_md`
imports `flax.nnx`, so 2 of `corpus/run_census.py`'s 8 public harnesses
cannot run on 0.11.1 until flax ships a fix. Measured: the same flax
imports fine on jax 0.11.0 with an otherwise identical package set.
**Nothing in stelling is pinned or skipped for it** — no CI lane imports
`flax.nnx`; the only lane that installs flax at all is
`acceptance-reproducer`, whose jaxfluids reaches `flax.linen` and 37 other
flax modules but nothing under `flax.nnx`, and whose exact 20-test
selection passes on 0.11.1.
`corpus/run_census.py` now attributes the failure to flax at its `jax_md`
import instead of surfacing a bare `AttributeError` naming a jax module.
(`corpus/interrogate_census.py`'s `_sqrt_defence_probe` is the tree's only
other `jax_md` import; it already prints its own failure, inside a section
about jax-md, and is left alone.)

**Cost of this bump:** one constant became a keyed map (and with it a test
split into the two findings it had been conflating, a report state, and a
canary state); one comment corrected from "series-stable" to "re-drive per
release"; one disclosure entry; one attributed import in the corpus. No
transfer, no analysis, and no verdict changed — so on the two columns this
file measures, the bump cost nothing. **What it actually cost is the
sampling unit**, and that is not a cost this file's tables can hold: every
future bump now has to be checked at release granularity rather than
series granularity, and there are far more releases than series.

## The rates — two, not one

**Name-level churn** — additive; one registry entry each; genuinely cheap.
Scales with the registry:

| bump | new prims | removed | lowering shifts |
|---|---|---|---|
| 0.10 → 0.11 | 1 (`unstack`) | 0 | `slice`+`squeeze` → `unstack` |
| 0.11.0 → 0.11.1 | 0 | 0 | none seen |

"New" here means **new to the census**, not new to jax: measured,
`jnp.unstack` traces to an `unstack` primitive on 0.10.2 as well. What
moved is blackjax's lowering, which is why the same row is filed as a
lowering shift. A registry gains an entry either way, so the cost is the
same — but the distinction matters for anyone reading this column as a
jax changelog.

**THE TWO ROWS ARE NOT THE SAME INSTRUMENT.** The 0.10 → 0.11 row is the
corpus census; the 0.11.0 → 0.11.1 row is the 70-form param census, because
the corpus census could not be run whole on 0.11.1 (Bump 2's ecosystem
paragraph). Its `0`s mean "the 70 forms reached the same 60 primitives on
both releases", not "the corpus did", and the two are not comparable as a
rate.

**Structural churn** — costs no registry entries and **invalidates
analyses**. Scales with the number of analyses stelling owns, so the
denominator is part of the measurement:

| bump | changes | analyses broken / owned |
|---|---|---|
| 0.10 → 0.11 | scan layout params removed; `Jaxpr`/`ClosedJaxpr` merged | **2 / ~2** (was recorded 1 / ~2) — the counter-vs-carry classifier reports `layout-unknown` on 0.11; and the `Jaxpr`/`ClosedJaxpr` merge broke the transfer-and-emission path on the OLD series in three places. The census/coverage counting survived |
| 0.11.0 → 0.11.1 | `reduce_max`/`reduce_min` gained `out_sharding`; the const-fold rule's source changed by one line | **0 / ~2** — no analysis reads either. The cost landed somewhere the columns of this table do not have: **query identity**, which moved for every harness containing a max/min reduction, and the tripwire's rule pin, which had to learn that a release is the key and a series is not |

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
libraries traced on a day-old jax release, which is a positive
observation about seven libraries on one release and is the only
quantitative reading anyone has bearing on the no-upper-caps rule. *It
read "the only quantitative evidence … and it points the right way",
which promotes one observation of a non-event into support for a
standing policy. One release is one draw; the rule is defended on its own
argument, and this observation has not contradicted it.*
