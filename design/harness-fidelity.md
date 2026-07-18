# The harness fidelity census — registered before the library-code comparison

**Status:** REGISTRATION, 2026-07-18. The criterion is **not new**: E2a's
permitted list registered *"transcribing the system as a jax function
faithful to the incident's own code"* before any harness existed
(`design/e2a-registration.md`, item 1). It was eyeballed and never
checked — exactly the state criterion (i) was in before it was
mechanized, and nobody called mechanizing (i) a renegotiation. This
census enforces it.

What **is** new — a silence filled, hence a **new registration** rather
than an amendment: the permitted-list text exists and is not wrong
(faithful transcription is permitted); the "voids the case" clause
enumerates the *tool's* jobs done by hand (bracketing, face
decomposition, interval evaluation, sign reasoning), never the system.
**The consequence of unfaithfulness was never stated.** Stated here,
before the census reads anything.

## Contamination and pre-commits, recorded

The analyst **wrote every harness under census** and knows their
contents — the same stake-in-the-answer shape the sampling registration
flags for the triage. What this registration fixes before the comparison
runs: the buckets, the consequence, and the judgement rule. The evidence
(library-code quotes at pinned refs) is gathered **after** this commit.

The proposer's pre-commit, from the work order, before any reading:
*"dfx#417 and dfx#207 look like traced user fields; the blackjax pair
look modelled; npy#249 unclear."* Scored in the reading.

## Buckets — fixed before reading

| bucket | meaning | counts? |
|---|---|---|
| **imported** | the harness imports and traces the library's actual code | **yes** |
| **user-MWE** | the harness transcribes the incident reporter's own system **verbatim** — same expressions, constants inline as filed. The MWE *is* the incident's own code | **yes** |
| **hand-transcribed** | written by hand with derivations or substitutions, matching the source (library code or user MWE) in **property-relevant substance**, every simplification listed and shown irrelevant to the obligation | **yes, with the fidelity disclosed** and a pointer to what it transcribes |
| **hand-modelled** | a stand-in the author invented; the discharging structure does not match any source | **cannot count 1.** A VERIFIED here is true about the model and meaningless about the incident — vacuity one level above the one criterion (i) catches. Permitted only as a labeled exhibit/demonstration, never as a counting case |

**Consequence (the filled silence):** a counted-1 case whose harness
reads hand-modelled is **voided, loudly** — enforcement of the
pre-registered criterion, not renegotiation; if a count moves, the
registration moved it. Sightings and trigger evidence originating from
hand-modelled harnesses are labeled as such wherever cited (forward
rule; no trigger text changes).

## The judgement rule — the hard line, fixed now

Judged **against the source's actual code at a pinned ref**, on
property-relevant substance: *the code path that discharges (or fails)
the obligation must exist in the source in the same form — same
variables, coupled the same way.* Simplifications are permitted only
where listed and shown irrelevant to the obligation. If the discharging
structure differs from the source's — a different variable, an invented
clamp, an invented update — it is **hand-modelled regardless of
plausibility**. The source line is **quoted** in the reading:
pointable-at, not asserted — the face-expression discipline,
transplanted.

## Two axes, not one

Fidelity × layer. A harness can trace real code and prove a property
about the wrong layer (dfx#417's discrete-step gap); a harness can
hand-model the right layer (bjx#D416). The layer classification cannot
catch hand-modelling; this census cannot catch layer gaps. Both ride
with any count.

## Reporting rule — third venue

**Wherever a count is reported, its fidelity breakdown rides in the same
sentence** — *"N mechanized: i imported, t hand-transcribed (disclosed),
0 hand-modelled"* — composing with the relation breakdown and the
denominator provenance chain, which already work this way.

## Bias and symmetry

This check can void cases that counted 1, and the proposer has been
arguing the results are weak — motivated in shape, declared. Defensible
because: **symmetric** (imported/user-MWE across the board would make
every count stronger than currently read); **conservative** (voids only
reduce counts and raise the bar); and **enforcement of a criterion
registered before any harness existed**.
