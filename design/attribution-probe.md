# The attribution probe — where did the effort go, registered before reading

**Status:** REGISTRATION, 2026-07-18. A new variable on the same 20
objects — the buckets do not move, the demand band does not move (same
grounds as the state/trajectory axis). Committed before any thread is
re-read.

## The derivable half (no probe needed, recorded here)

The tracker probe's registered terms were detection words — `nan`, `inf`,
`diverge`, `unstable`, `blow up`. You cannot search for `nan` and find a
failure nobody noticed. **The corpus is post-detection by construction:
detection cost ≈ 0 across all 20 hits, by selection.** This goes into the
tracker artifact as a scope note. What does *not* follow is that the
remaining cost is attribution rather than fix — that is the hypothesis
below.

## Definitions — fixed

| phase | definition |
|---|---|
| **Detection** | report → first notice. ≈0 by selection; not measured |
| **Attribution** | first report → the first comment that **correctly names the mechanism**, verified against what the fix actually did |
| **Fix** | that comment → the closing commit/PR |

Operationalization: the issue body counts as comment 0 — if the body
already names the mechanism, in-thread attribution is 0 and the case is
recorded as **named-in-body** (attribution paid pre-filing, off-tracker;
explicit narrowing statements in the body are its cost evidence).

Effort proxies (calendar time is not effort):

- **comment count per phase** — someone wrote each one;
- **wrong hypotheses aired** — mechanisms proposed and rejected;
- **recorded misdiagnosis** — a wrong answer **adopted** (acted on or
  accepted, not merely aired) before the right one. Binary; the sharpest
  signal. Archetype: the CFL case (float32 believed before CFL found).

## The duration-confound correction — recorded regardless of outcome

GitHub timestamps measure calendar, not effort: a 258-day thread can hold
four hours of work. Therefore the tracker artifact's "487d"/"258d" cost
signals are **calendar proxies**, and the demand band's cost evidence is
**weaker than reported**. This correction goes into the tracker artifact
whichever way this probe lands — it cuts against a band the proposer
likes, which is why it goes in.

## Bands — fixed

| finding | reading |
|---|---|
| **attribution dominates** — median attribution-comments/total > 0.7, or ≥8 of 20 with a recorded misdiagnosis | the cost is attribution; the product is diagnostic; every value hypothesis so far aimed at detection while the data was about diagnosis |
| **fix dominates** | **the reframe dies** — people knew what was wrong and it was still hard to fix; faster attribution saves nothing |
| **mixed** | the distribution, not the mean — one product per mode |

## Prior and bias, recorded (the proposer's, after four dead reframes)

> **Attribution dominates.** I want it to win; the definitions are tight
> for that reason. If the misdiagnosis count is 1–2 of 20, the reframe is
> dead and is reported as dead in the same sentence as the number.
