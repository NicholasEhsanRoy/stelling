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
