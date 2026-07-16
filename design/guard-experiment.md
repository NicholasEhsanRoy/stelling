# The guard experiment — defence-checking, registered before it runs

**Status:** pre-registration and hypothesis, 2026-07-17. Written after the
census interrogation (`design/census-interrogation.md`) and **before any
query has been run against any site**. The interpretability gate this
experiment required — can clamp vs. mask be drawn from IR? — was passed on
2026-07-17: of 19 wedge sites, 17 are clamps, 0 are pure masks, 1 is
clamp+mask with the shared-predicate nullification pattern visible in IR,
and 2 are unguarded-trivial. From IR alone.

## The hypothesis under test

> stelling's near-term value in **mature libraries** is checking defences,
> not finding bugs. In-bounds is a proxy, and a hand-written clamp makes
> the proxy true while leaving the failure intact: `where(idx < N, idx, 0)`
> guarantees in-bounds *by construction*, and the silent read of element 0
> is the same wrong-physics outcome the bug class describes — differing
> only in who wrote the clamp. The real property is one step upstream:
> **is the guard's predicate ever false over the region?** Dead → a true
> statement about the program nobody currently has. Fires → a witness
> where defended code silently reads the wrong element. Same integer
> reasoning, same solver, and `select_n` in the index cone finds the site.

The census supports the *setup*: 16+ defended sites vs. ~1 undefended, and
the defence pattern is not indexing-specific (the §5 probe: jax-md guards
its `sqrt(0)` backward-NaN hazard with the same hand-written shape —
`safe_mask` — at a completely different hazard class).

**Burden, acknowledged.** This reframe is made after a census that went
against the bug-finding claim, by the people who made that claim. It
therefore takes *more* obligations than the original, not fewer: its own
falsifier, its own corpus, its own buckets, registered here before
anything runs. It inherits none of the wedge experiment's evidence, and it
does not replace or modify the wedge's falsifier.

## Arguments against (recorded now so they cannot become post-hoc escapes)

1. **A dead guard may be a shrug.** Belt-and-braces is often deliberate;
   "provably dead over the documented region" is value only if authors
   want to know. Audience demand is unverified — zero users of the
   defence-checking kind have been sampled.
2. **Corpus circularity.** "Defences are everywhere" was measured in a
   corpus selected for maturity, by harnesses aimed at known suspects. The
   reframe generalises from the same seven targets the bug-finding claim
   struggled in.
3. **Site-counting is not value-counting.** Sixteen defence verdicts may
   be worth less than one Real bug.
4. **The V&V audience is presumed**, not sampled: the story that evidence
   of working defences is qualification-relevant is plausible and untested.

These are the same class of objections the wedge claim faces, one arm
over. The honest framing is a **portfolio**: bug-finding for the
undefended arm (research code — the value model's open re-registration
question), defence-checking for the defended arm (mature libraries). Each
arm carries its own falsifier; neither inherits the other's evidence.

## The experiment

**Corpus, fixed now:** the 17 clamp-guarded wedge sites of
`design/census-interrogation.md` §4 — diffrax ×6, jax-md ×11 (one of the
jax-md gathers also carries the shared-predicate mask) — plus the 2
unguarded-trivial sites as controls. No sites added after any query runs.

**Per-site property:** the guard predicate itself: `∀x ∈ R_site.
pred(x)` — *not* the in-bounds property the guard already enforces.
`R_site` is the site's input region, grounded in the library's own
documented valid-input contract, and **registered per-site before
solving**. Region choice is part of the registration: a region tuned
until an answer appears is the experiment dying.

**Outcome buckets, fixed now:**

| bucket | meaning |
|---|---|
| **Dead** | predicate provably true over `R_site`, and not *trivially* so (not provable by constant folding alone). Unreachable belt-and-braces: a real statement about the program |
| **Fires-by-design** | witness exists AND the site shows designed nullification — shared-predicate dataflow or sentinel provenance (jax-md's padded neighbor indices). Expected; not a finding |
| **Fires-unexplained** | a checked witness where the guard fires and no nullification is visible in IR: in-region, the code silently reads a fallback element with no visible discard. **The finding class** |
| **Unknown** | ⊤ in the chain, timeout, or no documented contract to ground `R_site` — sites without a groundable region are Unknown, never "region adjusted" |

**Falsifier, fixed now.** The defence-checking hypothesis over mature
libraries is **falsified** if, across all in-scope sites: **zero
Fires-unexplained AND fewer than 3 non-trivial Dead verdicts spanning at
least 2 targets.** (All-trivially-dead means the questions needed no
solver; all-Unknown means the machinery isn't ready; either way, no value
demonstrated.) Supported: ≥1 Fires-unexplained, or ≥3 non-trivial Dead
across ≥2 targets. In between: weak — publish the classification, do not
build on it.

**Anti-rationalisations, pre-registered:**

- Fires-by-design results do not count toward support, however many.
- The jax-md sentinel caveat is in the registration, not discovered later:
  its neighbor padding produces out-of-range indices deliberately; naive
  witnesses there are false positives and classify as Fires-by-design via
  provenance.
- The two while-carry sites (diffrax step-history buffers) enter as
  Unknown unless their regions can be honestly phrased over the carry — no
  invariant machinery may be invented mid-experiment to rescue them.

**What running requires:** the tier-0/1 registry (already unblocked), the
integer fragment for guard-predicate comparisons, and the per-site region
registrations. No general wedge; no scan/while descent (16 of 18
candidate sites are not loop-index-dependent).

**Relation to the standing decision:** this registration is an **input**
to the re-registration choice in `design/value-model.md` — it runs nothing
until that decision is made, and it does not make it.
