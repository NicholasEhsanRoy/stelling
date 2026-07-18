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

---

# Reading (2026-07-18 — every harness, against sources at pinned refs)

Sources: blackjax `e53f46b02f37`, diffrax `ae856adf9bb5`, numpyro
`aafb7d8cb306` (current HEADs; the incidents predate them — substance
judged on the discharging structure, which is stable across the span, and
noted where conditional).

| harness | case (count) | bucket | evidence — the source line, quoted |
|---|---|---|---|
| `e2a_hit386.py` | control (passed) | **user-MWE** | field expressions and constants verbatim from the filed MWE (`c = exp(a1) − x0`, …) |
| `e2a_417.py` | dfx#417 (1) | **hand-transcribed** (of the user's MWE) | drift `−2(x−u)` is the *exact* gradient of the posted quadratic `(x−u)²+(y−v)²`; disclosed substitutions: hand-differentiation (exact for a quadratic), tanh range → declared `[0,1]` bounds (sound over-approx), `sigma=0` from the thread's own args. **Counts, disclosed** |
| `cf_run.py::h_207` | dfx#207 (1) | **hand-transcribed** (of diffrax's clamp) | `diffrax/_step_size_controller/pid.py:555` — `dt = jnp.maximum(dt, self.dtmin)` under `if self.dtmin is not None:` (`force_dtmin: bool = True`). Same op, same variables, same coupling as my `jnp.maximum(dt_prop, DT_MIN)`. Simplified proposal factor listed and irrelevant (`max(x, DT_MIN) ≥ DT_MIN` for any `x`). **Disclosed conditionality: the clamp is the opt-in `dtmin=…` path** (default `None`). **Counts, disclosed** |
| `cf_run.py::h_249` | npy#249 (1) | **hand-transcribed** (of numpyro's exp) | `numpyro/infer/hmc_util.py:667–668` — `step_size = jnp.where(…, jnp.exp(log_step_size_avg), jnp.exp(log_step_size))`, followed at :672 by `jnp.clip(step_size, finfo.tiny, finfo.max)`. The discharging structure (exp, same variable) exists verbatim; the omitted clip is listed and only *strengthens* the real property (a float floor at `tiny` — conservative omission). **Counts, disclosed** |
| `cf_run.py::h_969` | bjx#969 (0) | **hand-modelled** | `blackjax/adaptation/mclmc_adaptation.py:438–442` — the NaN-branch select falls back to `(previous_state, step_size * reduced_step_size, 0.0)`: the cap is set from **`step_size`**, a *different variable*, coupled (`step_size_max ← step_size × 0.8`). Mine was a self-map `s_max × 0.8`. Different variable in the discharging structure → hand-modelled, per the rule |
| `cf_run.py::h_D416` | bjx#D416 (0) | **hand-modelled** | `blackjax/optimizers/dual_averaging.py:121` — `log_x = mu − (√step/γ)·avg_error` (prox-center shrinkage toward `mu`, averaged error). Mine was `log_eps − 0.1·h_stat`: invented update, no `mu`, no averaging, invented coefficient |
| `exhibit_632.py` | out of denominator | **hand-modelled** | self-declared: the real #632 mechanism (the `t1` clip) was never transcribed; exhibits are the permitted use of this bucket |

## Consequences

**No voids.** Every counted-1 case is user-MWE or hand-transcribed; the
bands stand — control-flow **Supported (2 of 4)**, E2a-with-control-flow
**3 of 7, Weak**. But the counts now carry their fidelity sentences:

> Control-flow: **"2 mechanized: 0 imported, 2 hand-transcribed
> (disclosed, quoted), 0 hand-modelled counting — and both non-mechanizing
> attempts were hand-modelled, so the experiment's *negative* evidence is
> about models."**
> E2a expanded frame: **"3 of 7: 0 imported, 3 hand-transcribed."**
> **Zero harnesses anywhere trace imported library code.**

## The h_969 knock-ons — the census's real finding

The faithful obligation is `step_size × 0.8 < step_size_max`:
**relational** — it needs the coupling `step_size ≤ step_size_max` (with
it, `0.8·step_size ≤ 0.8·s_max < s_max` discharges; without it, the
uncoupled product box makes the property **partially-false** — e.g.
`step_size=100, cap=0.01`). Posing the coupling is an `assume` — inert —
so the faithful case lands **blocked (inert assume)**, exactly where the
original E2a run classified bjx#969 *before* the control-flow harness
"made it posable" by modelling the coupling away.

1. **The dependency-shaped sighting was a model artifact.** The affine
   trigger's counting evidence drops to **0 valid** (+ 1 model-artifact,
   labeled; + the out-of-denominator exhibit). Trigger text unchanged.
2. **Affine's band-flip arithmetic deflates** — affine does not reach the
   faithful bjx#969 (that needs a constraining `assume` transfer, a
   different upgrade with its own registered vocabulary). Addendum in
   `design/affine-disclosure.md`.
3. **The control-flow census over-classified**: bjx#969 is not
   control-flow-only (it also needs the relational assume) — 4 → 3 of 6,
   and the **gate passes at exactly 3 of 6**. Correction in
   `design/control-flow-census.md`.
4. What a faithful bjx#D416 would do is **undetermined** — it needs the
   real DA body traced (imported), which today's harness API cannot pose
   (scalar `any_array` only; no pytree state declarations). Recorded as
   the instrument gap for imported-grade harnesses — a fact, not a
   trigger.

## The proposer's pre-commit, scored

"Blackjax pair look modelled" — **right, twice**. "npy#249 unclear" —
resolves to hand-transcribed. "dfx#417 looks like a traced user field" —
half: it is the user's field, hand-derived rather than traced.
"**dfx#207 looks like a traced user field**" — **miss**: there is no user
field in that harness at all; the chemistry ODE never appears — it is a
controller sketch (whose clamp is real diffrax code). Fifth
files-and-facts miss, by the proposer's own count.

## The workaround critique, confirmed

The blackjax pair were modelled because tracing the real loops was never
attempted (needs the library installed and its pytree state posed). The
workaround that made those cases posable manufactured both of the
experiment's negative findings — D416's partial violation and 969's
dependency shape — **both facts about models**. The critique ("the
workaround that makes a case posable is the one most likely to make it
verifiable") is confirmed in the contrapositive too: here the workaround
made the *failures*, and they were nearly built on.
