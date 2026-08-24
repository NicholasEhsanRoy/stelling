# The scatter-add / stack censused rows — build, soundness audit, fixes

**Status:** PASS RECORD, 2026-07-22. The row build that
`design/socket-decoupling.md` names as what the decoupling unblocks:
after these rows, sockets trace the real assembly instead of
transcribing it. Fresh blind builder; fresh adversarial auditor (these
rows can influence VERIFIED); `stelling.fidelity.gauge` as the
acceptance instrument — its first internal customer, exactly as
ordered ("you gauge new rows with the battery").

## What the rows are

- **scatter-add** (what `jax.ops.segment_sum` and `x.at[idx].add()`
  lower to): static, in-range, definitely-unique-or-default indices
  only; the defining accumulate semantic (duplicates each contribute)
  implemented in three places — the outward-rounded interval transfer,
  the per-element SMT emission (one addend per duplicate), and a new
  exact-Fraction **replay row** (the witness validator walks scatter-add
  now). Two measured jax 0.11.0 forms supported, everything else
  refused loudly (foreign dimension numbers, non-add combiners,
  out-of-range — mode-dependent, quoted — dynamic indices, and
  `unique_indices=True` with measured duplicates: the jaxpr's own
  promise makes that implementation-defined, so modeling it would be a
  guess; declined, the never-guess posture applied to the uniqueness
  promise).
- **stack**: pure element routing, exact both semantics, flag-OR under
  ieee.
- **ieee floor for scatter-add:** loud censused refusal for every form
  — the ℝ transfer's soundness rests on ℝ-associativity; float
  accumulate order is the backend's, not ours. No enclosure argument
  offered; cost is honest UNKNOWN under ieee, disclosed per equation.
- scatter-add is the **first registered transfer whose equation carries
  a sub-jaxpr** (the recorded combiner); its coverage accounting is now
  outcome-independent (below).

## The payoff (the reason this build exists)

Before: a library-free `segment_sum` normal-matrix assembly landed
`1 ⊤ (scatter-add ×1)`, obligations UNKNOWN, `escalation declined —
primitive 'scatter-add' is outside the supported emission set`.
After: **100% coverage, VERIFIED**, interval-definite where definite
and QF_NRA-discharged where straddling — and the hand-unrolled twin
(the old socket transcription pattern) produces identical
per-obligation statuses. Native and unrolled are interchangeable at
the verdict level; the next socket needs no transcription for
segment_sum-shaped assembly. Known reach limits, stated: the
`at[idx].add` array-index sugar propagates (after the static in-range
int64→int32 index-narrowing admission) but still declines at emission
on jax's index-normalization arithmetic; integer scatter-add emission
declines wholesale; int64 magnitudes past ulp>1 stay honestly
undecided (pinned as intended).

## The audit (fresh context; six findings, none UNSOUND)

The central question — does anything disagree about accumulate? — was
attacked from four sides at once: jax execution, transfer, emission and
replay **agreed on 1600+ duplicate-heavy samples**, and a REFUTED whose
witness is valid **only** under accumulate-all was replay-confirmed with
collapse-only points rejected. **The replay-confirmed REFUTED is the
reading; the four-way agreement is the size of the search.** Four
implementations that did not disagree have not resolved anything by
construction — they share the same accumulate reading, which is the
thing under test. Six findings, all instrument/posture-grade:

*Headed "zero UNSOUND, third round in a row" and opened "was resolved by
construction: four-way agreement" until 2026-08-24. A streak of quiet
rounds is a property of the rounds, and agreement among instruments that
share a premise is the weakest form of the strongest-sounding evidence.*

1. **F1 MISLEADING** — the coverage denominator dropped the recorded
   combiner equation on decline paths (real 9 vs ieee 8 on the same
   program; live counter vs `coverage.measure()` disagreeing). Fixed:
   declined sub-jaxprs count as unreached; totals are now
   dial-invariant and live==static (regression-pinned). The
   denominator is a function of the program, never of the outcome.
2. **F2 SHARP-EDGE** — `unique_indices=True` + measured duplicates was
   silently modeled as accumulate. Fixed: the promise-violation
   decline at both layers, quote pinned.
3. **F3 SHARP-EDGE** — the census hole: a demonstrated two-edit future
   misfiling (arithmetic transfer classified non-computing + ieee row)
   passed every assert and minted a false VERIFIED on int32 wrap.
   Fixed: probe-or-exempt classification censuses at both layers,
   every current primitive carrying a written reason, stale exemptions
   refused. **Banked as L22.** (The builder had self-reported the
   benign structural-set half of the same pattern.)
4. **F4 MISLEADING** — three auditor-invented mutations survived the
   gauge (single-group battery blindness; ungauged budget;
   emission-side mutations inexpressible). Fixed: multi-segment
   battery case, budget-boundary gate, a named emission seam making
   emission mutations expressible. The auditor's own extended battery
   now shows **every invented mutation caught, residual empty**;
   collapse-in-the-plan deepened 1→2 gates as mandated.
5. **F5 NOTE** — decline-text accuracy; the static in-range
   int64→int32 index-narrowing admitted as exact (boundary pinned both
   sides), so default-dtype sugar propagates.
6. **F6 NOTE** — the integer-exactness claim qualified by magnitude
   (ulp ≤ 1), the int64 2^62 bracket pinned as intended.

## Suite trajectory

968 → 992 (build) → **1003 passed** (venv-jax); 779+14 → 800+15 →
**809 passed + 15 skipped** (venv-nojax). No recorded verdict flipped
at any stage; probe and exhibits re-run clean; prose lint green
throughout.

## What this unblocks

The MADDENING socket (ARCHITECTURE.md's next-socket rule) built
against real traced code — a harness plus known answers, no
transcription, no per-socket fidelity stack for segment_sum-shaped
assembly. The gauge remains the instrument for whatever that build
still needs to bind.
