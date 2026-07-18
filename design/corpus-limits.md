# The input-declaration finding, and the trigger arithmetic — recorded, not built

**Status:** design note, 2026-07-18. Two findings and one decision, none
of which builds anything.

## Finding 1: the control-flow bottleneck is input declaration, not transfer functions

**Zero harnesses trace imported library code** — not one, across every
attempt (`design/harness-fidelity.md`). The census named why: `any_array`
is scalar/array-bounds-only; posing a real library function needs its
state declared, and the 17-of-20 finding means that state is **solver
infrastructure** — pytree carries (nested NamedTuples, mixed dtypes,
counters) and, wherever a kernel is in frame, **PRNG keys**, which have no
interval meaning at all. Key reuse was bjx#D416's actual cause: a faithful
blackjax harness can be unposeable for a reason that is not structural.

Evidence, four cases across three libraries: bjx#969 and bjx#D416
(hand-modelled because tracing the real loop was never attempted —
feasible-in-principle for D416's update body via per-leaf declaration
plumbing, untested), dfx#207 and npy#249 (posable only at the
property-relevant slice; their full bodies carry pytree state). The
control-flow transfers work where they were pointed
(`design/control-flow-hypothesis.md`); **they cannot be pointed at real
library code. More control flow does not fix that. Input declaration —
`any_pytree`, key handling, whatever it turns out to be — does.**

Not built. See the decision below for why not even a trigger is
registered today.

**The motivation control, registered now while it costs nothing
(2026-07-18, after the soundness audit):** the audit's clustering finding
has a build-time reading — every defect landed in the machinery built *to
make cases posable*, none in the core built before there was a hypothesis
to confirm. Speed was a variable; **so was stake** — dfx#207 is the
proof: a tautological pass that counted, built by someone who needed a
case to count. Motivated *building* leaves the same fingerprint as
motivated analysis, and it clustered exactly where the motivation was.
`any_pytree` (or whatever fills the input-declaration role), if ever
built, will be built under the strongest build-time motivation in the
project: it is the upgrade with the most evidence and the one that
unblocks the most cases, and the pull toward a version that makes cases
pose *and verify* will be maximal. Therefore, registered in advance:

> **`any_pytree` requires a fresh-context soundness audit as a build
> gate: the audit fires before any case it newly enables counts toward
> any band.** The same standard the control-flow machinery received —
> applied *before* a band depends on it rather than after one already
> did.

## Finding 2: this corpus cannot fire any trigger — the meta-finding

The counts, at threshold ≥ 2 each: **solver 0** (two experiments).
**Affine 0 valid** (the one sighting was a model artifact). **Sampling 1**
(bjx#D416, blackjax). **Constraining-`assume` 1** (bjx#969, blackjax —
both current single-sightings are single-source, which the ≥2-sources
default would also catch). The corpus is 7; three mechanized at peak, two
after the vacuity guard; **four failures with four distinct causes, one
each.**

That is not evidence that nothing needs building. **It is arithmetic: a
7-case corpus with 4 distinct failure modes structurally cannot fire a
≥ 2 threshold.** Applied to a corpus this small, the trigger discipline is
a permanent freeze — and a freeze is not a finding. The current state
cannot distinguish *"no upgrade is needed"* from *"the corpus is too small
to say"* — a fact about the **instrument**, the fourth time an instrument
has been the constraint (tracker terms, coverage counter, fidelity, now
corpus size).

**The expansions are banked and registered-adjacent already:** the search
terms `explosion` and `wrong results` (banked at the tracker probe, never
run); and **optimistix, lineax, equinox** — three libraries in the
primitive census corpus that were never tracker-probed. Running either
requires its own registration (terms, taxonomy, property test with all
four inherited corrections, bands) — it is not a footnote to this pass,
and it is not run here.

## The decision: no trigger for input declaration today — and why, in writing

The honest tension, argued now rather than resolved by drift: the upgrade
with the most evidence (imported-grade harnesses; four cases, three
libraries) has **no trigger**, because `any_array`'s scope was an MVP
choice nobody registered a trigger against. Registering one *now, knowing
the evidence,* is fitting — a trigger that can only fire on the data that
motivated it, which is the motivated-development shape with extra steps.
Registering none freezes the only thing the data points at.

**Decision: neither. The input-declaration trigger is registered *with*
the corpus-expansion registration, before the expansion's cases run.**
The expansion generates cases nobody has seen; a trigger written into its
registration is trigger-before-evidence on fresh data — the only clean
window. Until then, Finding 1 stands as a named candidate ("recorded, not
built"), this paragraph is the reason there is no trigger yet, and any
future attempt to build input declaration without that registration cites
this paragraph *against* itself. The freeze is therefore scoped: it ends
at the corpus expansion, not never.
