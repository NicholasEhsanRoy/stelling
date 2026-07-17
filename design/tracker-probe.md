# The tracker probe — long-horizon failures, registered before reading

**Status:** REGISTRATION, 2026-07-17. This section is committed **before a
single issue is opened**; the reading lands in a separate, later commit.
Nothing below may be adjusted after reading begins. Generated findings are
appended under "Reading" after registration is committed.

**The hypothesis being tested** (fourth candidate; the prior is 0-for-3):
classes with cheap *preventive* defences come back clean — confirmed twice
(index clamps, `safe_mask`; `check_grads` for custom-derivative rules).
Classes with only *detective* defences (a NaN check tells you the run died
at step 800k, not that it won't, nor for which inputs) should have a
target. Zero confirmations of that half so far. Long-horizon failures are
**loud** — they crash, get argued about, get filed — so the tracker is a
valid instrument here in a way it could not be for the silent wedge class,
and a null is readable rather than confounded.

## Corpus — fixed

| tracker | slug | discussions |
|---|---|---|
| diffrax | `patrick-kidger/diffrax` | not enabled |
| jax-md | `jax-md/jax-md` (canonical; `google/jax-md` redirects) | not enabled |
| jax-cfd | `google/jax-cfd` | not enabled |
| blackjax | `blackjax-devs/blackjax` | **enabled — searched** |
| numpyro | `pyro-ppl/numpyro` | not enabled (its forum is off-GitHub and **out of corpus**) |

Issues open **and** closed, `type:issue` (PRs excluded). Discussions where
the platform has them (blackjax only). Same libraries as the census, so
the artifacts are comparable.

## Search terms — fixed, with operationalization

| registered term | searched as |
|---|---|
| `nan` | `nan` |
| `inf` | `inf` |
| `diverge` | `diverge` |
| `unstable` | `unstable` |
| `instability` | `instability` |
| `blow up` | `"blow up"` (quoted phrase) |
| `blows up` | `"blows up"` (quoted phrase) |
| `drift` | `drift` |
| `long run` | `"long run"` (quoted phrase) |
| `after N steps` | `after steps` (word co-occurrence — N is not literal) |
| `works for small` | `"works for small"` (quoted phrase) |
| `intermittent` | `intermittent` |

Per-(repo, term) total hit counts are recorded **before any filtering**.
Retrieval: up to 50 results per (repo, term), GitHub best-match order;
the candidate set is the deduplicated union. Classification uses title,
body, labels, state, and thread metadata, with targeted comment reads
where needed. **If a term worth adding occurs mid-read, it is recorded
here and not used — it belongs to the next registration:**
*(none yet)*

## Taxonomy — every candidate lands in exactly one bucket

| bucket | meaning |
|---|---|
| **Long-horizon** | Fails only after many steps, or at large N, or intermittently, or seed-dependently. The target class |
| **Point-detectable** | Any single well-chosen test catches it; wrong at step 1. Not the class |
| **User error** | Genuine misuse: wrong argument, wrong config, wrong units. **Reserved for actual misuse** — *not* for "the maintainer said it wasn't a library bug" |
| **Performance** | Not the class |
| **Feature request** | Not the class |
| **Unclear** | Cannot tell from the thread. Reported as coverage, claimed as nothing |

**The counter-intuitive rule, fixed now:** issues closed as *not-a-bug*,
*your model is stiff*, *that's expected* are **the target population, not
noise**. A user hit a real long-horizon failure in their own code, loud
enough to file against a library, and the maintainer correctly said it
wasn't theirs — nobody's tool caught it. That is the class, seen through a
keyhole into the research-code arm the census could not reach. The default
instinct is to discard these; the registration forbids it.

## The property test — applied before the count is looked at

A Long-horizon hit counts **only** with a one-line constructive property:
*what property, over what region, would have turned this red before the
user hit it?* Circularity disqualifies ("the thing that went wrong doesn't
go wrong" is not a property; `state.rho > 0`, `isfinite(u)`,
`energy(s) < E_MAX` are).

## Cost signals — recorded separately, never folded into the count

Per hit: thread length, participant count, time-to-close, and explicit
cost statements ("took three days to narrow down", "lost a week of
compute"). **Pre-fixed:** a class with many instances and trivial cost is
not a target; a count with no cost signal is not a finding.

## The bands — fixed

| Long-horizon hits with a writable property | reading |
|---|---|
| **0–2** | **Falsified.** The class does not reach real code at a rate that matters, or is handled. Stop; the concept is sound and has no target; close the file. A zero is a stop, not a re-aim |
| **3–9, or ≥3 with no cost signal** | **Weak.** Real, rare or cheap. Publishable observation, not a sequencing argument |
| **≥10, in ≥3 trackers, with cost signal on ≥3** | **Supported.** Real, distributed, expensive, unreached by point methods. Licenses **only** the writing of a value model with its own corpus, experiment, and falsifier — not a build |

## The product split — required report

Every Long-horizon hit is also classified by shape: **safety-shaped**
(NaN/inf/crash/negative density/out-of-domain — universal properties, the
user supplies only a region; a tool) vs **accuracy-shaped** (drift,
conservation violation, "looked wrong after a while" — bespoke invariants
the user must write; a methodology). The ratio is reported alongside the
count and matters as much.

## Anti-rationalisations — pre-registered

- *"Closed as won't-fix, so it doesn't count."* It counts. It happened.
- *"It's really user error."* Decided by the taxonomy, not by whether the
  count is liked. A reclassification argument that first occurs after the
  count is visible is the forbidden move.
- *"The verifier probably wouldn't have caught this one either."* That is
  the property test's job, applied before the count is looked at.
- **A zero is a stop, not a re-aim.** If the loud class is absent from the
  loud instrument, that is the answer, and no fifth hypothesis gets
  written in the same breath as the fourth's obituary.
