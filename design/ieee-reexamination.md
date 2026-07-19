# IEEE re-examination of the counted verdicts — registration

**Status:** REGISTRATION, 2026-07-19, written and committed **before any
verdict is re-run under `ieee`**. The capability was recorded last pass
as newly-possible and deliberately unrun, requiring its own registration
(`design/ieee-semantics.md`, "Recorded, not run"). This is it.

Run under the **corrected** mode — the measured-FTZ+DAZ semantics with
the subnormal haze that the IEEE pass landed (ledger **L10**: model the
measured target, not the standard it claims) — never the pre-fix
IEEE-754 model.

## The fence (read this before reading any result)

The question is: **does each counted verdict survive under the
arithmetic a device actually runs?**

- A surviving verdict **upgrades the arithmetic the verdict speaks** —
  ℝ → device-float, the regulatory-relevant form.
- A surviving verdict **does NOT upgrade the relation to the incidents.**
  The E2a clause "bear on real recorded incidents" fails for a **layer**
  reason — 6 of 7 registered properties are controller/sampler
  adaptation state, and the mechanized cases are continuous-flow
  preconditions with the discrete-step gap named in their own stamps. No
  change of arithmetic moves a verdict to a different layer. **A green
  result here may not be banked against the incidents axis.**

## Two corrections to the commissioning premises (L4, before registering)

**(1) The corpus was misstated.** The work order names "(hit386,
dfx#417 — the real-mode VERIFIEDs)". `hit386` is the **positive control,
never a corpus member** (a correction already on file in
`design/mwe-census.md`). The counted verdicts are **dfx#417 and
npy#249** under the registered extended-ℝ test, and **dfx#417 alone**
under the finite-⊤ criterion the corpus expansion carries. Re-examining
"the counted verdicts" therefore **requires npy#249**, which the work
order omits. Corrected corpus below: both counted verdicts **plus** the
control (a control that fails under float is itself a first-order fact
about the instrument, and the registration's rule is that the control's
verdict is always reported).

**(2) The predicted live mechanism does not exist in the harness.** The
work order predicts "hit386's `exp` with a t-argument box reaching ~710
produces `inf`". Read from the harness source: every `exp` in hit386 is
`exp` of a **point parameter** — `a = [6.027, 4.412, 5.884, 3.674,
4.180]`, `b = −2.824` — never of the state. Their images lie in
`[0.059, 415]`. **Overflow via `exp` is unreachable.** The live
mechanism, if any, is different and is enumerated below; this correction
is exactly the discipline the work order asks for ("do not reason to one
live mechanism") applied to the work order's own mechanism.

## Corpus (fixed)

| harness | role | obligation shape |
|---|---|---|
| `corpus/supply/e2a_417.py` | **counted** (registered test and finite-⊤ criterion) | 4 face-flux inequalities on a linear drift over bounded boxes |
| `corpus/supply/cf_run.py` → `h_249` | **counted** (registered test only; voids under the finite-⊤ criterion) | `exp(log_eps − 0.05·grad) > 0` |
| `corpus/supply/e2a_hit386.py` | **positive control** (never a corpus member) | 3 face-flux inequalities on a rational/exp field with a **half-infinite** declared coordinate |

The `dfx#207` case is not in scope: it was **voided** by the ⊤-widening
guard as tautological and counts 0.

## Divergence classes, enumerated per obligation (not reasoned to one)

The ℝ-vs-float divergence classes, each shown filtered-or-flagged for
each harness. This enumeration is the registration's substance; the run
measures it.

**Class 1 — rounding collapse.** Filtered by construction for all three:
outward rounding is float-conservative (the `#632` exhibit measured
UNKNOWN, not a false VERIFIED), and the second audit narrowed the
residual to existence divergence, not rounding. Margins here are
order-1, not ulp-scale (dfx#417's face margin is 1.0; hit386's tightest
is `exp(a₀) − 415 ≈ −0.45`; npy#249's is `exp(·) ≥ 1.25e−9 > 0`).

**Class 2 — overflow → inf.** dfx#417: unreachable (bounded
add/sub/mul). npy#249: `exp` argument box is `[−20.5, 5.5]` → image
`[1.25e−9, 244.7]`, unreachable. hit386: **not via `exp`** (correction 2)
— but the declared `c ∈ [0.019, +∞)` is **half-infinite by
construction**, so `inf` is present in the declared set itself.

**Class 3 — invalid-op NaN (`0·∞`, `inf − ∞`, `0/0`).** dfx#417 and
npy#249: no `∞` reachable, so unreachable. **hit386: reachable, and this
is the live candidate.** Mechanically: the harness computes
`x0 = exp(a₁) − c` and the field recomputes `c = exp(a₁) − x0`;
interval propagation loses that correlation (dependency problem), giving
`c ∈ [0, +∞)` where the declaration said `[0.019, +∞)`. The product
`exp(a₃)·d·c` then multiplies an interval containing `0` by one
containing `+∞` — **exactly `0·∞`**, which is `0` under the stamped real
convention and **NaN under IEEE**. This is the failure mode the
`REAL_CONVENTION_ASSUMPTION` stamp line has named since the semantics
field was added; ieee mode is its first test.

**Class 4 — subnormal / FTZ.** All three out of band: dfx#417's values
lie in `[−3, 3]`; npy#249's `exp` image bottoms at `1.25e−9`
(≈ 10²⁹⁹ × MIN_NORMAL); hit386's declared `x₁ ∈ [6.8, 415]` and its
field values are order 10⁰–10⁴. **Nothing approaches 2⁻¹⁰²².** Predicted
filtered; the run confirms by observing no haze note.

**Class 5 — dtype.** All three declare `float64` under
`jax_enable_x64=True`; the binary64-only guard is not engaged. Predicted
filtered.

## Predictions (pre-committed, before any run)

- **dfx#417 survives.** No divergence class is reachable.
- **npy#249 survives.** Class 2/3/4 unreachable by the margins above.
- **hit386 is genuinely uncertain**, and the uncertainty is *not* about
  float behaviour of the program but about **flag hygiene in the tool**:
  the `0·∞` arises in the field's `dx0` component, which is **asserted
  only in obligation 3** — and obligation 3 evaluates the field at the
  *point* `c = C_MIN`, where no `∞` is present. Obligations 1 and 2
  assert the `dx1` component, which does not read `x0`/`c` at all. So
  hit386 survives **iff** the maybe-NaN flag is tracked per element and
  does not leak from the unasserted `dx0` to the asserted `dx1` through
  `jnp.array([dx0, dx1])` + index. Per-element hygiene was built and
  audited; this is its first contact with a real ∞-carrying harness.

## The reading distinction, fixed before it is needed

If a verdict fails to survive, the write-up must classify the cause:

- **genuine float divergence** — the program's own float execution makes
  the predicate false or indeterminate (a fact about the verdict);
- **artifact of our imprecision** — the NaN/∞ path exists only because
  interval propagation lost a correlation the declaration had (a fact
  about **the tool**, not the verdict). hit386's `c ∈ [0,∞)` recompute
  is this shape: the real `c` is bounded below by 0.019 and `0·∞` never
  occurs in any actual execution.

**These are different findings and must not be reported as one.** An
imprecision artifact does not make the ℝ verdict wrong; it makes the
ieee mode unable to reproduce it, which is a precision problem with a
named fix (relational/affine reasoning on the declared coordinate, or
emitting the declaration's own bound).

## Bands (fixed before reading; the continue decision is band-membership)

| result | band | continue? |
|---|---|---|
| **all three survive** — both counted verdicts and the control discharge under `ieee`, no unpredicted class fires | **clean** — the weak evidence strengthens toward device-arithmetic fidelity, with the relation to incidents explicitly unchanged | **continue to Task B** |
| **both counted verdicts survive; the control (hit386) does not** | **first-order finding** — either the ℝ verdict is not device-faithful, or the tool cannot reproduce it under float; the reading distinction above says which | **STOP, surface** |
| **any counted verdict fails** (dfx#417 or npy#249) | **first-order finding** — a counted verdict is ℝ-only; this changes what the counted numbers mean | **STOP, surface** |
| **anything else** — an unpredicted divergence class fires, a verdict flips in an unforeseen direction, a real-mode verdict moves, or a run raises | out of band by definition | **STOP, surface** |

**No band is amended after reading.** A surprise is a stop by
construction.

## Invariants this run must not disturb

Every harness is re-run under **`real`** first: all three must reproduce
their recorded statuses byte-identically. The `ieee` run is additive and
uses a separate mode; if any real-mode verdict moves, that is the
fourth-row stop.
