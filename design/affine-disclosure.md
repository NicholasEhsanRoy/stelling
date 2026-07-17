# Affine and the band it would cross — disclosed before any build

**Status:** DISCLOSURE, 2026-07-18. Registered before any affine work, so
that a coincidence knowable now cannot be discovered conveniently later.
Nothing here licenses a build; the affine trigger has **not** fired
(`design/control-flow-hypothesis.md`: 1 dependency-shaped UNKNOWN, not 2).

## The arithmetic, computed after the control-flow run

E2a, re-read on its 7 in-semantics hits with the control-flow transfers
now built, mechanizes **3** — dfx#417 (`design/e2a-run.md`), dfx#207 and
npy#249 (`design/control-flow-hypothesis.md`), same 7-hit denominator,
expanded frame. The value model's E2a band is **≥ 4**
(`design/value-model-v2.md`). **One short.**

The one dependency-shaped UNKNOWN, bjx#969, is exactly what affine forms
reach. If affine mechanized it, E2a would be **4 of 7 — crossing Weak into
Supported, on the value model's own registered falsifier.** (Not 3→5: only
bjx#969 is affine-reachable; bjx#D416 is a genuine violation affine cannot
fix.)

## The disclosure

> Affine, if it ever becomes licensed and is built, is the upgrade that
> would plausibly move E2a from 3/7 (Weak) to 4/7 (Supported) — crossing
> the value model's own band on the strength of a single hit. **This was
> foreseen and disclosed before any affine work.** Building the machinery
> that flips one's own falsifier is the shape of motivated development,
> whatever a trigger says; the only thing that makes it legitimate is the
> order — trigger first, on evidence that predates the arithmetic. **The
> affine trigger has not fired.** If it later fires and E2a is re-run
> post-affine and lands Supported, this paragraph is cited in the same
> sentence as the count.

Same shape as "supported with an all-precondition breakdown is weaker than
the word suggests" — stated before the data, where it is disclosure rather
than an excuse.

## The protection was an accident — for the second time

The **IEEE-first precondition** on the affine trigger
(`design/unknown-triage.md`) was added for soundness, from an unrelated
argument (the dfx#632 straddle). It happens to mean the band-flipping
upgrade **cannot be built cheaply**: float semantics ships first, it is
expensive, and it flips nothing. A friction on the motivated path,
arriving from somewhere else entirely.

Twice now a protection has been **accidental rather than chosen** — the
#632 straddle (the tool's imprecision coinciding with float's), and this
(IEEE-first gating the band-flip). **Both times a reasonable optimisation
would have deleted it**: tighten the domain, and the straddle closes and
the false green appears; drop the "unnecessary" IEEE precondition, and the
band-flip becomes cheap.

## The general question, worth a design note not a probe

> **What else is currently safe for reasons nobody chose, that a sensible
> improvement would remove?**

This is the general form of the #632 finding, and the project has no habit
of asking it. Two instances found by accident is not a sample; it is a
prompt. Candidates to examine when convenient (recorded, not investigated):
outward rounding's conservatism on every monotone comparison (a tighter
domain removes it); the `blocked`/inert-`assume` conservatism (a real
`assume` transfer removes the shielding); the frame-unposeability that
kept 6 of 7 hits from ever reaching a checkable obligation (control flow
removed some of it — and the E2a count rose as a result). Each is a place
where making the tool *better* along one axis could make a verdict *less*
safe along another, unless the coupling is written down first.
