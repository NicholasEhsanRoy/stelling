# The semantics-drift probe — registered before reading a single release note

**Status:** REGISTRATION, 2026-07-18. The question: is `safe_mask`'s
trigger — *jax changed the behaviour of `sqrt` at 0 and silently
invalidated downstream correctness* — a **class** or a story about one
event? Committed before any changelog entry is read.

## The instrument correction, stated up front

Trackers cannot answer this: **silent means not filed** — a tracker
measures loud upgrade breakages and is structurally blind to the silent
ones (the wedge's confound, relocated). A near-zero from a tracker grep
would be uninterpretable. **The primary instrument is JAX's own
changelog**, which documents semantics changes whether or not anyone
downstream noticed — the property that makes this probe falsifiable.

Three instruments, ranked:

1. **JAX's changelog** — the surface: how many semantics changes exist.
2. **Downstream reaction** — per counted silent change: did any corpus
   library touch its guards near that release (git `-S` in a date
   window), or file/receive tracker traffic? A reaction means loud; no
   reaction means irrelevant **or** silent — not separable, and read as
   such.
3. **Tracker grep, demoted to loud base rate**: terms `"after upgrading"`,
   `"worked in"`, `"jax version"` across the five corpus trackers, counts
   recorded, nothing more claimed.

## Corpus and span — fixed

`CHANGELOG.md` from `jax-ml/jax` (fetched 2026-07-18; 210 release
headings, jax 0.1.58 / Jan 2020 → 0.11.0 / July 2026 — the file **does**
reach the 2020 `sqrt` era).

**Primary counted span: 0.4.0 → 0.11.0.** Rationale, fixed now: 0.4 was
the consolidation release; every corpus library's floor sits inside this
span, so it is the span within which any *currently shipped* guard was
calibrated. **Secondary, outside the count:** the 2020-03 window is read
only to date the `sqrt`-at-0 trigger — the motivating instance predates
the counted span and is reported as such, not counted in the bands.

**Method:** every release in the span: "Breaking changes" sections read
in full; "Changes" / "Bug fixes" / "Deprecations" sections triaged by
behaviour-change phrasing ("now returns", "no longer", "instead of",
"default", "promotion", "precision", "rounding", "gradient", "now
raises", …) with every hit read in surrounding context. Bug fixes are
in scope: an upstream bug fix that changes numeric output is a silent
semantics change downstream.

## Taxonomy — what counts, fixed

| class | meaning | counts? |
|---|---|---|
| API change | rename, signature change | no — breaks at import, loud by construction |
| Deprecation | warned, then removed | no — loud |
| New feature | invalidates nothing | no |
| **Semantics change — loud** | same call, same signature, different behaviour, and the difference announces itself (raises, warns, NaNs) | counted separately |
| **Semantics change — silent** | same call, same signature, **a different number and no signal** | **the target** |

Per silent change, the crux question, answered in one line each:

> Would a downstream guard calibrated to the old behaviour still hold
> under the new one? (Yes → real and harmless. No → an instance.)

## Bands — fixed

| finding | reading |
|---|---|
| **≤2 silent** semantics changes in the span | **Falsified.** JAX is semantically stable; 2020 was an outlier; close it |
| **≥3 silent, with loud downstream reactions on ≥3** | **Self-correcting.** The ecosystem notices; a verifier adds latency, not detection. Weak |
| **≥3 silent, ≥1 guard-invalidating, few or no reactions** | **The surface is real.** Licenses a value-model input and nothing else |

## The circularity limit — pre-stated so it cannot arrive as a discovery

**stelling's transfer functions are its model of jax's semantics.** If
jax changes `sqrt`'s backward behaviour, stelling's `sqrt` rule is stale
by the same commit, and it would cheerfully prove a downstream guard
unnecessary under its own out-of-date model. **The naive proposition is
circular.** It survives only in the **differential** form: versioned
transfer functions, verdicts stamped with their series, and a cross-bump
diff reporting *which guards changed status*. That is exactly what the
treadmill costs — the cost now sits on the other side of the ledger. If
this probe returns supported, this limit returns with it.

## Bookkeeping — recorded, not used here

**`comment-grade`** is added to the bucket vocabulary for future
archaeology-type registrations: a failure narrative present in source
comments but absent from the commit message (the equinox DAZ case). Kept
recorded-not-used, the same way banked search terms are handled.

---

# Reading (2026-07-18 — after the registration commit)

Method as registered: all Breaking-changes sections in the span read in
full; Changes/Bug-fixes/Deprecations triaged by behaviour phrasing (138
candidate lines), every hit classified, ambiguous ones read in context.

## Silent semantics changes in 0.4.0 → 0.11.0: **13** (changelog lower bound)

With the crux — *would a guard calibrated to the old behaviour still
hold?* — answered per line:

| change | release | crux |
|---|---|---|
| **`jnp.empty`/`empty_like` now truly uninitialized (was zeros)** | 0.11.0 | **invalidating** — anything relying on zero-init silently reads garbage |
| **`rng_bit_generator` under vmap draws only from the first key** | 0.4.26 | **invalidating** — batched-independence assumptions silently violated |
| **`dctn`/`idctn` axes default fixed to match SciPy** | 0.10.0 | **invalidating** — transforms run over different axes; entirely different numbers |
| **`ceil`/`floor`/`trunc` keep integer dtype (was float upcast)** | 0.4.31 | **invalidating** — downstream arithmetic changes class (int division) |
| **`ldexp` gradient corrected** | 0.4.34 | **invalidating for gradient guards** — the sqrt-at-0 shape: upstream fix, downstream numbers move |
| `randint` distribution de-biased (8/16-bit) | 0.7.2 | statistical guards move; config escape existed |
| `arange(step=…)` no longer computed on host | 0.9.2 | exact-grid users invalidated (the #657 family); slack guards hold |
| `threefry_partitionable` default → new PRNG streams | 0.5.0 | draw-calibrated tests invalidated; statistical guards hold |
| complex `sign` → `x/abs(x)` | 0.4.24 | complex-branch guards move |
| `logsumexp(return_sign=True)` complex convention | 0.4.24 | same family |
| complex `geomspace` branch choice | 0.4.26 | same family |
| `cov` single-row matches NumPy 2.2 | 0.8.0 | narrow, silent value change |
| `arr.view(dtype=None)` returns array unchanged (was float cast) | 0.7.2 | narrow, silent return change |

Loud semantics changes (announce via raise/NaN), counted separately: ~6
(empty-`cov` NaN, SciPy-matching NaN for negative ints, triangular-solve
inf/nan fix, scalar-`initial` error, `bool(empty)` error, non-scalar
conversion error). API changes and deprecations: excluded as registered.

**Rate: ≈3.6 silent semantics changes per year**, as a lower bound —
because the probe's sharpest secondary finding is that **the 2020
`sqrt`-at-0 trigger is absent from the changelog entirely** (the only
sqrt match in 210 releases is `sqrtm`'s addition). The primary instrument
undercounts its own target class; every count above is a floor.

## Downstream reactions (instrument 2, bounded)

- `rng_bit_generator`, `ldexp`, `dctn`: **zero uses in the corpus
  libraries** — irrelevant-or-silent, not separable, recorded as such.
- `threefry` default: the one **loud** case — upstream-orchestrated
  migration with a dedicated update note. Not ecosystem self-correction;
  upstream shepherding.
- **`jnp.empty`: four live non-test call sites in equinox** (the
  ecosystem's substrate), found by this probe two days after the change
  shipped. The batch-norm pair is the exhibit: EMA state buffers created
  with `jnp.empty`, guarded by a first-time flag — under ≤0.10 a
  mishandled guard was masked by zero-init; under 0.11.0 the same failure
  blends uninitialized memory. **The drift class does not merely
  invalidate guards; it mints new guard obligations** — a freshly created
  L3 question, dated to this week. (Whether these specific guards hold
  was not assessed; that is exactly the L3 experiment's shape.)

## Loud base rate (instrument 3, demoted as registered)

`"jax version"` appears in 203 threads across the five trackers (numpyro
81, diffrax 57, blackjax 39, jax-md 20, jax-cfd 6); explicit
`"after upgrading"`/`"worked in"` phrasings: 15. The loud channel is
busy — and structurally disjoint from everything counted above.

## Band: **the surface is real**

≥3 silent (13), ≥1 guard-invalidating (at least five), reactions few and
mostly upstream-driven. Per the registration this **licenses a
value-model input and nothing else** — and it arrives welded to the
pre-stated circularity limit: stelling's transfer functions are its model
of jax's semantics, so the proposition survives only in **differential
form** (versioned transfers, series-stamped verdicts, cross-bump
guard-status diffs). The treadmill's third speed now has a number: ~3.6
per year, floor, measured against an instrument that misses its own
motivating instance.
