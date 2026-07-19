# The regional obligation — the third framing, registered before it runs

**Status:** REGISTRATION, 2026-07-19, committed **before the harness is
written or run**. MIME pinned `7ce1efb`. A **usefulness test on held-out
MIME code: it never enters E2a**, produces no count, and moves no
standing figure.

## Why this exists

`design/private-track-criterion.md` scored F1 and F2 against the private
criterion and **both failed clause (iii)** — the scar mesh's
`cos ≈ 0.11` *violates* the alignment floor that F1 and F2 both take as a
**precondition**, so neither verdict fires on the failing mesh. Closing
F1→F2 yields a code-level result that is still not a CI check; the
assume-emission build failed on its own gate.

The same application surfaced a third framing, banked then and registered
now: **make the mesh-quality region the *subject* of the obligation
instead of its precondition.** The scar's `cos ≈ 0.11` is *inside* the
region the meshes actually span, so a verdict over that region can fire
exactly where F1/F2 are silent. This is the shape the heat job's **A2
control** already had (an obligation over the node's own validated
regime, not over its safe envelope) — used there as a control, never as a
product.

A premise correction from the same reading rides with this: **the
`cos ≥ 0.71` one-liner does not exist.** `0.71` appears nowhere in the
FVM package and there is no mesh-quality check of any kind, so **at the
pinned ref the scar is caught by nothing.** That makes the regional
obligation the *only* candidate for catching it, not one of two.

## The obligation, and the controlled comparison

Quantity: the over-relaxed orthogonal coefficient
`E_f = |Sf|² / (Sf·d)` (`operators.py:251-253`, introduced by the scar's
own fix `91e95e6`).

**Region (the change), owner-supplied and labelled as such:** the
cylinder mesh reaches `min cos(Sf, d) ≈ 0.11`; the declared cos region is
therefore **`[0.11, 1.0]`** — the range the scar mesh actually spans.
This characterisation is not on disk (it was supplied in the FVM job's
work order and recorded there as owner-supplied).

**Everything else is held identical to F1**, so the *only* difference
between F1 and R1 is the cos region and the absence of the floor as a
precondition: `|Sf| ∈ [0.5, 2.0]`, `|d| ∈ [0.5, 2.0]`, bound `B = 8.0`.
Round numbers, disclosed as choices, inherited unchanged.

**Two posings, paralleling F1/F2 exactly:**

- **R1 — the lemma form** (F1's polar substitution `Sf·d = |Sf||d|cos`
  and the exact `a²/a` cancellation, both previously disclosed):
  `∀ a ∈ [0.5,2], dm ∈ [0.5,2], c ∈ [0.11,1]: a/(dm·c) ≤ 8`.
  **No `assume`.** Parallels F1; scores clause (i) the same way F1 does.
- **R2 — the code's own raw form** (F2's transcription: `Sf`, `d` as
  independent component boxes, the dot product written out), obligation
  `area²/(Sf·d) ≤ 8`, **with the floor neither assumed nor imposed** —
  the region enters through the declarations alone. Parallels F2; scores
  clause (i) the way F2 does.

Both are run interval-only **and** with solver escalation, and both
verdicts are reported.

## Ring discipline (as with F1/F2/F3)

A **fresh-context transcriber** writes the harnesses. It sees **FVM
source only**; it does **not** see the validation results, the
characterisation study, `known_anomalies.yaml`, git history, this
registration's bands, or **any statement of which verdict is wanted**.
The region and the bound travel as **bare numbers**. The scar provenance
is the main agent's and stays there. A transcriber that knows "fires on
the scar" is hoped-for will, with no bad intent, transcribe toward it.

This is a **run, not a build**: the existing tool on a new obligation. No
new transfers, no new domain, no builder/auditor cycle.

## Predictions (pre-committed, before the harness exists)

- **R1 interval-only: UNKNOWN, partially-false shaped.** The quotient
  over the region spans roughly `[0.25, 36.4]` against the bound `8`; it
  is genuinely true at high cos and genuinely false at low cos, so no
  definite verdict is available to interval propagation and **no
  `assume` is present, hence no DROPPED note** — cleanly distinguishable
  from the dependency wall by the tool's own diagnosis.
- **R1 escalated: REFUTED with a witness at low cos.** The fragment is
  QF_NRA; the divisor `dm·c` has interval `[0.055, 2.0]`, which
  definitely excludes 0, so division is emittable and the escalation is
  not declined. The negation is satisfiable (e.g. `a=2, dm=0.5,
  c=0.11 → 36.4 > 8`), so the expected result is **sat**, replayed for
  box membership and violation, yielding a concrete
  mesh-quality counterexample.
- **R2: dependency-wall UNKNOWN.** `Sf·d` over independent component
  boxes straddles 0, so the division crosses zero and widens to ⊤;
  escalation additionally declines on the possibly-zero divisor. Same
  wall F2 hit, reached by a different route (no `assume` is involved
  here at all — this is the dependency problem, not the relational-assume
  problem).

If these hold, the fork the work order names is answered precisely:
**R1 fires without any build but fails clause (i); R2 satisfies clause
(i) but cannot fire.**

## Distinguishing the two UNKNOWNs — mechanical, fixed now

- **dependency-wall**: a `DROPPED`/inert note, or a ⊤ arising from a
  zero-crossing divisor / declined transfer, with the coverage line
  naming it. UNKNOWN is then a **precision** verdict, not a found
  instability.
- **partially-false**: no such note; the obligation's own interval
  straddles the bound with both sides genuinely attained, and the
  escalation (if it runs) returns sat with a replayed witness.

If the diagnosis is genuinely ambiguous, that is a **must-stop**.

## Bands — every row surfaces (this measurement gates the next move)

| verdict | reading | action |
|---|---|---|
| **REFUTED-with-witness, or clean partially-false** | **fires on the scar** — the regional framing scores (iii) where F1/F2 fail; a CI-shaped candidate that needs **no build** | **STOP, surface** |
| **dependency-wall UNKNOWN** | the regional obligation hits the wall — the machinery would be justified by (iii) this time, not by the failed F1→F2 argument; **resurrects assume-emission, re-scoped** | **STOP, surface** |
| **VERIFIED** | contradicts the scar → the transcription or the obligation is mis-specified | **STOP, surface** |
| **ambiguous diagnosis** | the judgement call that separates CI-check from wall | **STOP, surface** |

There is no clean-continue row: the terminal action is **register,
measure, report, stop**. No band is amended after reading.

## Four-clause scoring, to be applied to whatever comes back

Per `design/private-track-criterion.md`: **(i)** code's own form;
**(ii)** CI time; **(iii)** fires on the scar — the clause the regional
framing exists for; **(iv)** recurring value — the argument *to test, not
assume*, is that it re-derives the property against the actual code and
region each run, where a frozen threshold (which does not exist here)
would go stale on code change. Scores recorded for R1 and R2 separately,
since they differ on (i) by construction.

## Non-goals

No CI wiring (a live mesh-statistic pipeline is a separate human
decision — this measures only whether the check *fires*). No build; if
the wall is hit, that surfaces as a re-justified build for a later
human-in-loop pass. No count; MIME held out. No new transfers, domain,
or emission surface.

---

# Reading (2026-07-19 — registration `0f3b6b2` preceded every number below)

`corpus/supply/mime_fvm_regional.py` (blind transcriber; every result
below independently re-measured by the main agent with its own harnesses
and its own `Fraction` arithmetic).

## Results — all four predictions held

| run | verdict | diagnosis |
|---|---|---|
| **R1 interval-only** | **UNKNOWN** | **no propagation notes, 100% coverage** — partially-false shaped, mechanically *not* the wall |
| **R1 escalated** | **REFUTED, witness-backed** | cvc5 (QF_NRA primary, 58 ms) and z3 (12 ms) **both sat, agreeing**; witness replayed |
| **R2 interval-only** | UNKNOWN | 19/21 known (⊤ from `sqrt`, `integer_pow` in the numerator) |
| **R2 escalated** | UNKNOWN, **declined** | `'div': divisor may be zero over the declared box — SMT-LIB2 division is underspecified at 0`; **zero invocations** |

**The witness, verbatim and verified:** `a = 3/2`, `dm = 1`,
`cos = 1/8 = 0.125` → `a/(dm·c) = 12 > 8`. Independently confirmed in
exact rationals with no stelling code in the loop: **in the declared
region**, **violating the bound**, and **`cos = 0.125` is below the 0.71
stable floor and inside `[0.11, 1.0]`** — a low-alignment point, which is
the regime the scar mesh occupies.

**The controlled check (mine, not in the spec): the region is the sole
cause.** The *identical* R1 form over F1's region `cos ∈ [0.71, 1.0]`
still **VERIFIES**. Everything else — boxes, bound, form, semantics — is
held fixed. So the flip from VERIFIED to REFUTED is attributable to the
region and nothing else.

**R2's wall, isolated.** The transcriber's own side-measurement (R2b:
`|Sf|² = Sf·Sf` by hand, removing both ⊤s) and my independent
reconstruction both land **UNKNOWN at 100% coverage with the identical
decline** — so the ⊤s were incidental. The wall is the divisor:
`assert(Sf·d > 0)` over independent component boxes is itself
**unknown**. This is the **dependency** problem, not the
relational-assume problem.

## Band adjudication — R1 and R2 land in different rows; both stop

- **R1 → row 1** ("REFUTED-with-witness"): **fires on the scar regime**,
  needs no build.
- **R2 → row 2** ("dependency-wall UNKNOWN").

Both rows are STOP-and-surface, so the terminal action is unchanged. The
fork registered in advance resolved **exactly as predicted**: *R1 fires
without any build but fails clause (i); R2 satisfies clause (i) but
cannot fire.*

**Correction to row 2's editorial gloss (the action is unchanged).** The
band said a wall here would "resurrect assume-emission, re-scoped."
**The measured cause does not support that consequence:** R2 contains
**no `assume` at all** — the floor is neither assumed nor imposed, by
construction — so assume-emission is not what is blocked. What is blocked
is a divisor that straddles zero over decorrelated component boxes, which
points at **affine/relational domains** (keeping `Sf·d` correlated) or a
multiplicative reformulation, neither of which is the assume-emission
build. Recorded as a correction to the band's causal attribution, not to
its action; **assume-emission remains dead on its clause-(iii) failure**
and is not resurrected by this result.

## Four-clause scoring (per `design/private-track-criterion.md`)

| clause | **R1** | **R2** |
|---|---|---|
| (i) code's own form | **FAIL** — polar form, two disclosed derivations (the transcriber flagged this unprompted: the REFUTED is about the polar form, "not about line 251 as executed") | **PASS** — the raw form, as written |
| (ii) CI time | **PASS** — 7 equations, ~70 ms of solver time | PASS (would be) |
| (iii) **fires on the scar** | **PASS** — first time on the private track: a refutation over the geometry region the failing mesh occupies, with a concrete low-cos counterexample | **cannot fire** — walled |
| (iv) recurring value | **not established** — R1 is regression-on-change like F1, and per-run value additionally needs a live mesh statistic (out of scope here) | n/a |

**Neither clears all four**, so the headline the work order reserved for
a four-clause pass — "the first genuinely CI-shaped result" — **is not
earned**, and I am not claiming it.

**What *is* earned, stated exactly:** **clause (iii) passes for the first
time in the project's history.** F1 and F2 are silent on the failing
regime by construction; R1 produces a sound refutation over it, with a
witness naming the alignment value, using **no new machinery** — existing
interval propagation plus the solver leg built two passes ago. The
solver's first demonstrated customer was a conditioning obligation; this
is its **first use that bears on a real failure the author paid for**.

## Banked observations — not acted on

1. **The tool can invert the question.** Bisecting the declared region
   floor, interval propagation flips VERIFIED→UNKNOWN at
   **`cos ≈ 0.500000`**, matching the exact algebra
   `a_hi/(dm_lo·B) = 2.0/(0.5·8) = 0.5`. So instead of *"is the
   coefficient bounded given the floor?"* the tool can answer *"what
   floor does a given bound require?"* — 40 propagations, sub-second,
   design-time. **Caveat that must ride with it:** `B = 8.0` was itself
   chosen in F1 with `0.71` in mind (~42% margin over the corner value
   5.63), so the proximity of the derived `0.5` to the owner's empirical
   *divergent* boundary (`≤ 0.59`) is **suggestive, not independent**.
2. **Per-run value needs one input the project doesn't have**: the actual
   cos range of the meshes under test. That is the CI-wiring question,
   and it is a human decision, not a verification one.
3. **Nonvacuity was UNCHECKED on all four runs** (bare numbers, no
   membership conditions declared). For R1 this is materially repaired by
   the witness itself: a replayed in-box counterexample **constructively
   demonstrates the region is inhabited**, which is the property
   nonvacuity exists to establish. Verified in my own arithmetic.

## Ring discipline — verified, not asserted

The harness contains no reference to the scar, the divergence, the
cylinder mesh, the anomalies file, commit `91e95e6`, the validation
results, or the `0.59` divergent boundary. Its only "commit" mention is
the transcriber explicitly **disclaiming** that it verified the pin
(which was out of its scope). The region and bound travelled as bare
numbers, and the transcriber flagged, unprompted and against interest,
that R1's refutation is about the polar form rather than the executed
line — the caveat that becomes its clause (i) failure.

## Held

No count; MIME held out from E2a. No CI wiring. No build. Every band row
surfaces, so: **report and wait.**

---

# Part A — the reading challenged, and materially corrected (2026-07-19)

Commissioned to challenge the unclaimed-headline reading. **Conclusion:
the headline stays unclaimed — but two of my stated reasons were wrong,
and the measurement found a bounded path to earning it that I did not
know existed.** Everything below is measured.

## Correction 1 — R2's wall is not "the dependency problem" (my error, published)

I wrote that R2's UNKNOWN was the dependency problem and that it "points
at affine/relational domains." **Both are wrong.** `Sf` and `d` are
declared **independent**, so there is no correlation to lose. The
declared region genuinely contains degenerate geometry: an anti-aligned
pair `Sf = (1,0,0)`, `d = (−1,0,0)` lies inside R2's box, and
`assert(Sf·d > 0)` over it is **`violated-over-set`** — definitely
false, measured. So R2's region contains sign changes and a pole
(`Sf·d = 0`), the obligation is genuinely ill-posed there, and **the
tool's UNKNOWN and its division decline were both honest.** The defect
was the region, not the domain.

## Correction 2 — the real dependency is a *self*-correlation, and F1's "derivation" was doing precision work

The code computes `area² / (Sf·d)` where `area = |Sf|`, so **`a` appears
on both sides**. Measured over **F1's own safe region** `cos ∈ [0.71,1]`:

| form | interval | escalated |
|---|---|---|
| `a/(dm·c)` — F1's, cancellation done by hand | **VERIFIED** | VERIFIED |
| `a·a/(a·dm·c)` — the code's shape, uncancelled | **UNKNOWN** | **VERIFIED** |

**F1's second disclosed derivation — the `a²/a` cancellation — is not
presentational. It is the precision device that makes the obligation
interval-provable at all.** Without it, even the safe region is
undecidable by intervals. I had read those derivations as convenience;
they were load-bearing.

## Correction 3 — the solver already dissolves it; what remains is plumbing

The same uncancelled, code-shaped form under escalation:

- safe region `[0.71, 1]` → **VERIFIED**;
- failing region `[0.11, 1]` → **REFUTED with witness**
  `a=1, dm=1, cos = 15/128 ≈ 0.117` — *closer to the scar's 0.11 than
  R1's own witness.*

So **the mathematical obstacle to clause (i) does not exist.** The
solver handles the self-correlation natively, which is precisely what it
is for.

What actually blocks a code-form check today is **three ordinary,
censusable rows**, measured by attribution:

1. **`reduce_sum` unregistered** — `jnp.sum(mesh.Sf * mesh.d, axis=1)`
   (line 250, verbatim) falls to ⊤, taking the divisor with it;
2. **`integer_pow` unregistered** — `mesh.area ** 2` (line 251,
   verbatim) falls to ⊤;
3. **`slice` outside the SMT emission set** — with the sum written
   longhand the propagation reaches **100% coverage**, and escalation
   then declines with `primitive 'slice' is outside the supported
   emission set` (vector indexing).

Two transfer rows and one emission row. None is a domain problem, none
is assume-emission, none is affine.

## Position on the commissioned openings

1. **Underselling clause (iii)?** No — I hold the reading. It was stated
   as the earned result and it survives; the new measurement strengthens
   it (a code-shaped form refutes with a witness at `cos ≈ 0.117`).
2. **Can R1+R2 clear it as a pair?** **No, and for a sharper reason than
   I gave:** their regions are not merely different, R2's *contains
   poles and sign reversals that R1's excludes by construction*, and the
   constraint that would align them (`|Sf| ≥ 0.5`, `cos ≥ 0.11`) is not
   expressible in a box — it needs relational assumes. The pair route is
   closed. The single-obligation route (trace the code's expression over
   a polar-parameterised region) is open but blocked by the three rows.
3. **Is clause (i) right?** **More clearly right than when it was
   written, and I argue it independent of the payoff.** Correction 2 is
   the argument: a hand-derived reformulation was *silently doing
   precision work*, and if line 251 changed shape the cancellation could
   become invalid with nothing to notice. That is exactly the staleness
   (i) exists to catch. **I am not relaxing it** — it still fails today,
   and I have named what would make it pass rather than redefining pass.
4. **Is the inversion the right next thread?** It is cheap and real, but
   I now rank the **three-row build** above it: it converts an unclaimed
   headline into a claimable one on a real failure, on the code's own
   form. That is a **build decision** — surfaced, not executed.
5. Factual corrections: the two above, both against my own reading.

## Sighting bookkeeping, corrected (2026-07-19)

Correction 1's consequence for the demand ledgers, stated because the
two builds it touches are different builds:

- **R2 is not a new sighting for anything.** Its UNKNOWN is a region
  defect, not a domain limitation.
- **A "third affine sighting" was never committed** — a search of
  `design/` finds no such claim (the only "third sighting" on file is the
  elimination-mode one, `bjx#D416`, a different and still-unadopted
  thread). So there is nothing to retract; the pre-Part-A line in this
  file that said the blocker "points at affine/relational domains" is
  corrected above, in place.
- **The standing counts, unchanged by this pass, and kept distinct:**
  **affine** has two demand sightings, both on held-out code — heat's
  stencil and LBM's Mach bound — with F3's predicted-behind-the-wall
  sighting explicitly *not* advancing it, and **valid counting evidence
  0**. **Relational / constraining-assume** separately has two —
  `bjx#969` (corpus-side) and F2 (real-solver-side). These are **not the
  same pair**: a linear-correlation domain and a relational-assume
  capability are different builds, and F2/`bjx#969` belong to the
  latter.

## Consequence

Per the work order's gate — *"if it changes materially, stop and surface
before Part B"* — **Part B was not registered or run.** The inversion
registration waits on a human read of the corrections above and of the
build question they raise.
