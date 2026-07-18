# The second bill — registered before the classification is made

**Status:** REGISTRATION, 2026-07-18. The `any_pytree` probe showed the
sugar takes a case from *can't-declare-the-state* to *declared, and 44%
of the program is ⊤*. **Posed is not mechanized, and the gap is ordinary
transfer work, per target** — the census finding one level up: the
primitive census billed the ecosystem; the **second bill** — per-target
array-semantics and registry completeness — only comes due on real
state. This document estimates it on the clean case before any build.

**Contamination, recorded:** the clean case's ⊤ list (19 equations, 11
primitives) is already on disk and in context from the probe. What this
registration fixes before the judgement is made: the buckets, the
majority criterion, and the consequence of each outcome. The per-⊤-eqn
facts (operand dtypes/shapes, params) are extracted mechanically so every
bucket assignment is pointable-at, not asserted.

## Buckets — fixed

| bucket | meaning |
|---|---|
| **trivial** | monotone or structural; a three-line transfer (`abs`, `eq`/`ne`, logical ops, `reshape`, `stop_gradient`, weak-type/literal-source converts, `pow` on a known-sign base with a decline guard) |
| **array-semantics** | the genuinely-hard class the audits already bled on: rank broadcasting, batched/scalar selectors, axis reductions — anything where the interval meaning depends on shape |
| **out-of-ℝ** | needs a semantics the tool doesn't have (float-exact, …) — banked, never built from here |

**Majority criterion:** by ⊤-equation count (primary), by distinct
primitive (secondary, reported alongside). **The reading, fixed now:**

- **Mostly trivial** → the build is **bounded**: `any_pytree` + a finite
  registry list reaches diffrax; the fresh-context builder gets that list.
- **Mostly array-semantics** → `any_pytree` reaches diffrax *in
  principle* only; the real work was never the sugar, it is the class
  that already produced false-VERIFIEDs twice, and it must be named as a
  different, larger build **before** it starts. **The (A)-vs-(B) fork
  reopens** — "reaches diffrax" gated on open-ended hard-transfer work is
  much closer to (B) than the sugar implied. Said plainly if that is
  where the number lands.

## §2's boundary probe — registered with it

The key cone's *why* generalises: a `PRNGKeyArray` is unposeable because
its entire semantic content is that its bits are uniform and its derived
outputs independent — and interval abstraction's defining move is
discarding the correlations independence is made of. **The dependency
problem, total instead of partial.** The boundary is therefore not
"samplers": it is *any property whose truth lives in a correlation
intervals discard* — and a sampler property that holds for **any** bits
may pose fine.

**One case, fixed now:** the strongest key-independent blackjax candidate
— the MCLMC **isokinetic momentum-norm invariant** (`‖p‖ = 1` after the
update, for any key: the integrator normalizes by construction). Posed as
`Σ pᵢ² ≤ 1 + ε` on the real kernel's output. No count. Outcomes: if it
poses and its obligation does not route through a discarded correlation,
the walled half is smaller than "all samplers"; if it hits a wall, name
**which** — key-decorrelation, partial dependency, or registry — with the
blocking structure quoted.

---

# Reading (2026-07-18 — `corpus/supply/second_bill.py`)

## §1: the clean case's 19 ⊤, classified from extracted facts

Every ⊤ equation's operands/params were printed mechanically; the
assignments below cite them.

| primitive (⊤ eqns) | facts | bucket |
|---|---|---|
| `abs` ×4 | `float64[1]`/`float64[]` | **trivial** (monotone-piecewise) |
| `eq` ×3, `ne` ×1 | scalar float vs literal | **trivial** (three-valued, the lt/le machinery) |
| `or` ×2, `and` ×1 | `bool[]` | **trivial** (three-valued logic) |
| `stop_gradient` ×2 | identity | **trivial** |
| `reshape` ×1 | `[1] → ()`, `dimensions=None` | **trivial** (structural, data-preserving) |
| `pow` ×1 | scalar base, **literal exponent** (the PID coefficient) | **trivial** with a base-sign decline guard |
| `convert_element_type` ×2 | `new_dtype=float64`, weak-type flip / literal src | **trivial** (value-preserving forms the rule doesn't yet name) |
| `select_n` ×1 | scalar `which`, `(1,)` cases (declined) | **array-semantics** (batched/scalar selectors) |
| `reduce_or` ×1 | `bool[1]`, `axes=(0,)` | **array-semantics** (axis reductions — degenerate here, but the class is shape-dependent) |

> **Tally: trivial 17 of 19 equations (89%); 9 of 11 primitives.
> Array-semantics: 2 of 19. Out-of-ℝ: 0.**

**Reading, per the registered criterion: MOSTLY TRIVIAL → the build is
bounded.** `any_pytree` + a finite registry list reaches diffrax; the
fork does **not** reopen. The fresh-context builder's list, finite and
closed: the nine trivial rows above, plus three array-semantics items —
batched/scalar `select_n`, axis reductions, and (from the hard case's
declines) rank broadcasting. The array-semantics items inherit the §3
guard rule below and land in the builder's audit scope: they are the
class that produced false-VERIFIEDs twice.

## §2: the boundary probe — the candidate escapes the key wall and lands on the partial-dependency wall

The momentum-norm harness **POSED** (392 eqns, 56% known, obligation
unknown). Attribution, walls in the order the obligation meets them:

1. **Registry** (`reduce_sum ×18`, `integer_pow ×7`, `sqrt ×15` — the
   norm pipeline itself): the second bill, ordinary.
2. **Partial dependency**: behind the registry sits the structure that
   *makes* the invariant true — blackjax `integrators.py:466`,
   `new_momentum_normalized, _ = _normalized_flatten_array(new_momentum_raw)`
   — an explicit per-step renormalization, i.e. `p/‖p‖`, a
   self-correlation intervals discard. Carrying `‖p‖ = 1` through the
   update needs exactly the correlation the domain forgets — **bjx#969's
   shape, in the invariant's own maintenance**.
3. The **key cone** is upstream of the momentum's *values* but **not of
   the property's truth** — the invariant holds for any bits, which is
   what made it the right probe.

**So the principle is confirmed by instance, and the boundary is
relabeled:** the corpus splits by **whether the property's content
survives interval abstraction** — keys are the 100%-dependency form,
normalizations/ratchets the partial form — not by library. The sampler
half is *not uniformly key-walled*: this property's wall is partial
dependency (relational/affine territory, its own registered trigger,
never a patch) behind ordinary registry rows. The
"`any_pytree`-reachable / key-blocked" split from the target probe is a
**rough proxy** for the real boundary and is so marked there.

## §4, the decision inputs, assembled

- §1: **bounded** — mostly-trivial, finite list. The fork stays closed.
- §2: the reachable half does **not** grow today (the escape-candidate
  lands on the partial-dependency wall), but the walled half's *reason*
  is now named per-case rather than per-library — which the corpus
  expansion inherits: sampler hits get attributed to key / partial /
  registry walls individually, not written off as a bloc.

Per the fixed readings: **build `any_pytree` — fresh-context builder, the
finite list, §3's guard rule, the registered audit gate, scoped to the
array-state half.** The decision to exercise the license is the
maintainer's; this document's job was the two numbers, and they are 89%
and one-named-wall.

## Carve-out (2026-07-18, before the build): the convert forms are not trivial

The two `convert_element_type` declines were bucketed trivial. **Wrong
bucket, corrected before the builder exists:** they are additions to the
`_t_convert` whitelist — the exact artifact the first audit built
*because value-changing casts were the most reachable false-VERIFIED*
(finding 1), and whose widening under fix pressure produced finding 4-B
at ±2³¹. Widening a soundness boundary is not a registry row.

- **The two forms leave the trivial bucket and the builder's list
  entirely.** Corrected tally: **trivial 15/19 (79%), array-semantics
  2/19, whitelist-widening 2/19, out-of-ℝ 0** — still mostly-trivial by
  the registered criterion; **the bounded reading and the closed fork
  stand.**
- **The whitelist rule, registered:** a `convert` form enters the
  whitelist **only with a witnessed regression test — a real jax trace
  showing the cast changes no value on the declared domain** — and each
  entry goes through the audit gate **flagged as a whitelist widening**,
  never folded into a general review. Weak-type flips need the witness
  like anything else: "weak type" is a jax concept, not a soundness
  guarantee.
- **The builder's list is therefore: eight trivial rows** (`abs`,
  `eq`/`ne`, `and`/`or`, `stop_gradient`, `reshape`, `pow` with a
  base-sign decline guard) **+ three array-semantics items**
  (batched/scalar `select_n`, axis reductions — `reduce_or` as observed —
  and rank broadcasting). Convert stays exactly as audited.
