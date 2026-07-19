# IEEE semantics — the second dial position, censused not complete

**Status:** PRE-BUILD registration 2026-07-18; build record follows.
Ledger review (first application of the standing practice,
`design/lessons-ledger.md`): **L8** (this build makes the registered
dial real and the affine ordering mechanical), **L1** (the exactness
lift ships here), **L5** (guard rule spec-carried), **L2** (conjunctive
adjudications state both faces), **L6** (audit aimed where acceptance
can't see), **L4** (the commissioning order itself carried two recorded
corrections — the IEEE under-scoping admission and the L1 direction
inversion).

## Scope — the censused float behaviours, nothing more

The job is the behaviours the known false-VERIFIEDs need, already
enumerated by the marker tests: **rounding collapse** (`fl(t+dt)` can
equal `t`), **overflow to `inf` as a value** (the real domain's
`0·∞ = 0` convention is unsound under IEEE), and **NaN** (production by
`0·inf`/`inf−inf`-class operations; all comparisons false under NaN, so
NaN never helps discharge and always falsifies a comparison
obligation). A full IEEE-754 domain (subnormal subtleties, signed-zero
distinctions, non-default rounding modes) is NOT the job; anything
outside the census stays ⊤-with-maybe-NaN under IEEE with the gap
noted.

**The mechanism that makes this cheap and exact:** Python floats ARE
IEEE binary64 with round-to-nearest. Under IEEE semantics the semantic
value of an operation *is* the float result, so for the monotone
arithmetic core the float image over a box is bracketed by native float
endpoint arithmetic **exactly** — no outward rounding (that existed to
bracket the *real* value). Point inputs give point float outputs:
`t + dt > t` with sub-ulp point `dt` becomes **definitely false**
(REFUTED), the strongest honest answer to the #632 shape; boxes
spanning the collapse boundary straddle honestly; `dt ≥ ulp(t)` over
the whole box discharges — float-modelling, not blanket refusal.
libm-backed ops (`exp`) keep 1-ulp outward brackets under the faithful
rounding assumption, sound for float too.

Every registered transfer is censused for IEEE: verified sound as-is,
given an IEEE variant, or declined to ⊤-maybe-NaN with the gap quoted —
no silent reuse (the real-mode `0·∞ = 0` inside `mul` is the proof that
blind reuse is unsound).

## The dial

`real` stays default and byte-identical — every recorded verdict is a
real-mode verdict and remains valid as such. `ieee` is additive and
stamped (`SEMANTICS_IEEE`; the `0·∞ = 0` assumption rides only in real
mode; IEEE mode stamps its own endpoint-arithmetic assumption).

**Two mechanical guards land with the dial:**
1. **Affine's IEEE-first precondition becomes enforceable** (registered
   in `design/unknown-triage.md`; the accidental-protection finding):
   propagation gains the domain selection point (`domain="interval"`,
   the only registered value); a tightened domain under `real`
   semantics refuses with the rationale quoted, and a can't-drift test
   pins the refusal. Affine, when built, lands against this guard.
2. **IEEE-mode propagation refuses solver escalation** (a seam the
   work order did not name, added at registration): the SMT backends
   emit over Reals — escalating a float-semantics obligation would
   prove the ℝ obligation under an IEEE-stamped claim, a
   semantics-mismatch false VERIFIED. Escalation declines under
   `semantics="ieee"` with the reason quoted (QF_FP-style encodings are
   a future, separately-censused fragment), double-guarded per the
   one-invariant-two-mechanisms commitment.

## Acceptance — the markers were built for exactly this event

The two marker tests pinned at the second audit ("must consciously flip
if the semantics dial ever moves") are the pre-built acceptance case:
- `(x+x)·0` over an overflow-reaching box: discharges in real (ℝ says
  0), must NOT discharge under IEEE (inf·0 is NaN-possible);
- `r ≤ +∞` over a ⊤ loop output: discharges in real, must NOT under
  IEEE (⊤ is maybe-NaN; NaN ≤ ∞ is false).
Plus the #632 construction: `t + dt > t`, sub-ulp `dt` — never VERIFIED
under IEEE (point form: REFUTED; box form: UNKNOWN), while the
genuinely-safe `dt ≥ ulp(t)` variant still discharges. Verified
independently by the main agent before the audit; **the markers' flip
is the acceptance criterion, their non-flip the failure.**

## Verdict integrity

No real-mode verdict may flip: the recorded set (hit386, dfx#417,
#632 exhibit, cf quartet, solver acceptance, MIME F-set) re-runs
byte-identical under the default. The lift of the exactness
certification (ledger L1) out of the assume machinery into a shared
primitive is behavior-preserving.

## Recorded, not run: the re-examination this opens

IEEE mode makes it possible to ask whether the *counted* verdicts
survive float semantics (is hit386's box IEEE-faithful or only
ℝ-faithful — the question a device that runs float actually asks).
**Not run this pass**: re-checking counted cases under a stricter
semantics is its own registration, filed before it is read. Recorded
here as newly possible, nothing more.

---

# Build record (2026-07-19)

## The census, as implemented

33 registered transfers censused: **10 sound-as-is** (structural / data
movement, with flag routing), **23 given ieee variants**, **0 whole
transfers silently reused**. Unmodellable *configurations* decline with
the gap quoted rather than being modelled: non-binary64 arithmetic,
`pow`/`convert`/`scatter`/`gather` on flagged inputs, every pre-existing
form decline now landing ⊤-maybe-NaN. Non-registry constructs (`cond`,
transparent wrappers, `assume`, literals, the ⊤ fallback) censused
separately. Two jax 0.11.0 measurements shaped the result: `pow(NaN,0)`
is `1` (forcing pow's flagged-operand decline) and `lax.max/min`
propagate NaN.

The native kernels (`ieee_add/sub/mul/div`) do endpoint arithmetic in
binary64 with **no outward rounding** — under IEEE semantics the float
result *is* the semantic value — so point inputs give point outputs and
the collapse becomes decidable rather than merely straddling.

## Acceptance, verified independently (both faces, per ledger L2)

Every conjunctive claim checked on both faces by the coordinator's own
harness before the audit launched:

| shape | real face | ieee face |
|---|---|---|
| `(x+x)·0`, overflow-reaching box | VERIFIED (ℝ says 0) | **does not discharge** (inf·0 NaN-possible) |
| `r ≤ +∞` over a ⊤ quantity | VERIFIED | **does not discharge** (⊤ is maybe-NaN) |
| `t + dt > t`, point sub-ulp `dt` | — | **REFUTED** (`fl(t+dt) == t`) |
| `t + dt > t`, box spanning the collapse | — | **UNKNOWN** |
| `t + dt > t`, `dt ≥ ulp(t)` over the box | — | **VERIFIED** |

The last row is the one that matters most: the mode **models** float
rather than refusing everything. The marker tests were built to flip on
exactly this event; their flip is the acceptance, their non-flip would
have been the failure. Stamps checked both ways too: ieee never carries
the `0·∞ = 0` assumption; real carries it unchanged.

## The audit — float-behaviour fidelity, and the target that isn't a standard

**Round 1: 1 UNSOUND / 1 FRAGILE / 2 COSMETIC.**

**U1 — the mode modelled IEEE-754; the target doesn't.** Measured jax
0.11.0 CPU binary64 is **FTZ+DAZ**: subnormals are flushed in
arithmetic, *in comparisons*, and in libm; eager matches jit; no flag
disables it. The mode modelled textbook gradual underflow, so seven
end-to-end shapes contradicted the measured execution *at the declared
point* — including `assert(x·x > 0)` at `x = 1e-160` as a false
VERIFIED, `assert(x > 0)` at `5e-324` as a false VERIFIED with no
arithmetic involved (DAZ reaches comparisons), and `t − dt < t` at the
underflow boundary: **the project's own diffrax bug shape, reappearing
in the mode built to catch it**. Scope measured precisely: 30 000 point
trials, every divergence inside the subnormal band, zero outside.

This is **SOUNDNESS.md's "one jaxpr, three devices, three numerics"
commitment biting a second time, and harder** — the first bite was a
disclosure gap (precision config in the stamp); this one was a
soundness bug. Registered as ledger **L10**: a semantics mode must
model the *measured target*, not the standard the target claims.

The fix is a **subnormal haze**: any interval meeting the open band
`(−MIN_NORMAL, MIN_NORMAL)\{0}` is hulled with `0`, applied at the
kernels (operands pre-corner — which routes DAZ-created `0/0` and
`0·inf` into the NaN flag — and results post), at comparison operands,
and at declarations/consts/literals on entry. The mode is now sound for
**both** semantics, flushing and gradual, because subnormal-band
outcomes become indeterminate; a stamped assumption discloses this and
its measured basis. Adjacent channel closed by the builder in the same
round: strict (`gt`/`lt`) assume certification under ieee now requires a
flush-robust witness, since a band-only strict region is empty at
runtime under DAZ.

**F1** — `_ieee_select_n` dropped the selector's maybe-NaN flag on an
*unenforced* invariant (the comment stated it; nothing checked it);
guarded. **C1/C2** — the ieee solver-absence reason still described
outward-rounded arithmetic; the reverse mode-mix pairing (real
propagation + ieee refusal escalation) was ungated. Both fixed; the
mispairing gate is now symmetric in semantics.

**Re-attack (standing rule): the haze fix was dtype-blind — U2.**
`MIN_NORMAL = 2⁻¹⁰²²` defines the band in binary64 terms, but the
measured flush is **per dtype**: float32 subnormals (`|x| < 2⁻¹²⁶`) are
perfectly normal f64 numbers, invisible to the haze, and this target
flushes them too (`float32(1e-45) > 0` is `False`;
`convert(f32 1e-45 → f64)` is `0.0` — so the `_EXACT_CONVERSIONS`
whitelist's "value-preserving for every representable source value" was
measurably false under DAZ). Four end-to-end faces, two false VERIFIEDs
and a wrong REFUTED among them.

Fixed by **completing the binary64-only guard rather than modelling
per-dtype flush** (decline-don't-mismodel): every ieee comparison and
every non-f64-float convert source declines with the gap quoted; real
mode keeps the whitelist byte-identically (the ℝ reading is about
represented values, and the real stamp disclaims float execution). The
builder's own sweep then caught **the same shape one level up, again**:
the assume classifier consumed comparison *equations* directly, bypassing
the guarded transfer, so an f32-band comparison could narrow, certify a
DAZ-empty precondition, or raise a *false harness-defect* claim
(`UnsatisfiableAssumptionError` on a comparison that is TRUE at
runtime). Non-f64-float comparisons now drop inert under ieee — no
narrowing, no satisfiability claim, no raise. **Second re-attack: 0
UNSOUND / 0 FRAGILE / 0 COSMETIC**, across residual-channel sweeps (all
17 value-judging transfers decline f32), the false-harness-defect faces,
over-decline controls, and a fresh real-mode byte-comparison against
`HEAD`.

**The recurring shape, worth naming:** three times in this pass an
invariant was enforced at one consumer and assumed at the others
(the select_n selector, the convert whitelist, the assume classifier).
Recorded against ledger **L7** — enforcement belongs at the choke point,
not at each consumer.

## Verdict integrity

**No `real`-mode verdict changed anywhere.** The full recorded set
re-runs identical (hit386 VERIFIED / mutation REFUTED; dfx#417 VERIFIED
/ mutation UNKNOWN; the #632 exhibit UNKNOWN; the control-flow quartet
VERIFIED/VERIFIED/UNKNOWN/UNKNOWN; MIME F1 VERIFIED, F2 UNKNOWN, F3
UNKNOWN), the pre-existing test baseline passes unmodified (only the
marker comments were edited, to point at their ieee companions), and the
auditor independently byte-compared real-mode behaviour against `HEAD`
across a 17-query battery twice. Suites: **venv-jax 522 / venv-nojax
445 + 7 skipped / venv-solverfree 514 + 8 skipped** (baselines
431 / 354+7 / 423+8).

The `#632` exhibit deserves one line of its own: under `real` it stays
UNKNOWN (its recorded status, and the honest one for ℝ), while the same
obligation posed under `ieee` at a point is now **REFUTED** — the tool
can finally decide, in the semantics where the bug lived, the exact
predicate that was green upstream for 258 days.

## Standing after this build

- **IEEE is a selectable, stamped mode**; `real` remains the default and
  every counted verdict remains a real-mode verdict.
- **Affine's precondition is now mechanical, not merely registered** —
  the `domain` dial refuses any tightened domain under `real`, with the
  rationale in the refusal text and a can't-drift test on it. Affine,
  whenever built, lands against this guard.
- **The exactness certification is a shared primitive**
  (`stelling/exactness.py`, ledger L1) rather than assume-machinery
  internals — the LA contract inherits it instead of rediscovering the
  hole.
- **Recorded, not run:** whether the *counted* verdicts survive float
  semantics is now an answerable question. It is not answered here;
  re-checking counted cases under a stricter semantics is its own
  registration, filed before it is read.
