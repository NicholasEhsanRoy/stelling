<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Proposed decline messages — **APPLIED** (all five sections)

**Status: APPLIED.** All five numbered sections below shipped. Each carries an
**Applied** line naming the test that pins it, and
`tests/test_doc_examples.py::test_this_page_s_numbered_sections_each_name_a_live_pinning_test`
requires every one of those tests to exist and to name this page back — so a
section cannot lose its pin without going red.

This page kept a `PROPOSED, NOT APPLIED` header through the changes that
applied it, which is the defect `proposed-declaration-dtype-check.md` records
and names: a claim divergence on a DOCUMENT. Corrected here rather than
quietly retitled, because the divergence is the point.

**Every "Today" block below is kept as it was measured**, and every "Proposed"
block as it was drafted. They are the argument — a message-design proposal
with its before/after pairs rewritten to the shipped text records nothing —
and where the shipped text departs from the draft, the section says so rather
than editing the draft to match.

Evidence: two independently-blinded agents wrote contracts against `jax_md` and
`jaxfluids`, neither told anything about this project's conclusions. **They named
the same worst message, the same best message, and the same fix.** The fix
already exists in the tool — one decline does it right and the others don't.

## The pattern to propagate

The `div` guard, rated **9/10** by one agent and the model for the other:

```
escalation declined: 'div': divisor may be zero over the declared box — SMT-LIB2
division is underspecified at 0 (element 0 spans [-69.4444444444446, 69.44444444444463])
```

> *"It names the primitive, the reason, **and prints the offending interval**. The
> interval is what made it actionable."*

**Three properties: names the primitive, gives the reason, prints the box.**

---

## 1. The worst message — silent UNKNOWN

**Today:**
```
assert #0: unknown — undecided for 2/2 element(s)
```
> *"This message told me nothing."* — 1/10
> *"This is where I'd have quit if I hadn't been asked to keep going."*

Both agents hit it. In one case the whole cause was `0.5 * x` losing an exact
zero endpoint; coverage was 100%, there were no ⊤, and no note.

> **That particular instance no longer arises**, and the record above is kept
> as it was measured rather than rewritten: audit 0.2.0 M16 gave `mul` the
> exact-rational route `add` and `div` already had, so `0.5 * x` over `[0, 4]`
> is now exactly `[0.0, 2.0]` and that obligation discharges. The message
> design below is unaffected — an endpoint can still miss a bound by one ulp
> wherever a transfer must *bracket* (`exp`, `pow`, `sqrt`), and that is where
> the shipped sentence is exercised now.

**Proposed:**
```
assert #0: unknown — undecided for 2/2 element(s).
  The obligation's operand spans [-5e-324, 8.0]; the asserted bound is 0.0.
  Coverage is complete (7/7 equations known, no ⊤), so this is a PRECISION
  result, not a coverage one: the interval is wider than the property needs.
  The lower endpoint misses by 5e-324 — one ulp — which is outward rounding,
  not a false property. Common causes: a repeated variable (interval
  arithmetic cannot see cancellation), or an exactly-stated threshold.
```
**The single highest-value line is the operand's interval**, which the verdict
already knows. One agent: *"printing the final interval would have taken me from
40 minutes to 40 seconds."*

> **Applied.** The undecided line now carries the operand's span, the asserted
> bound, and — the §2 rule below — where the box came from. Pinned by
> `tests/test_div_straddle_decline.py`. Measured, on a divisor declared over
> `(-1.0, 1.0)`:
>
> ```
> assert #0: unknown — undecided for 1/1 element(s); the operand spans [-inf, inf]
> and the asserted bound is operand <= 1000000000.0 — lhs is stelling's own ⊤ from
> 'div' at <path>:10 (h) (its interval transfer declined this form), not a
> declaration-derived range
> ```

---

## 2. The most misleading message — `sqrt` blaming the user

**Today** (abridged; the real one is ~90 words):
```
'sqrt' declined this form: sqrt has a real value only for a nonnegative argument,
and the argument interval's lower bound -inf is negative: the obligation arg >= 0
is not established over the declared box, so the box includes out-of-domain
points (jnp.sqrt of a negative is NaN). Declined — the pow domain posture (a
domain-restricted transfer refuses the out-of-domain box loudly, and the
propagator turns the refusal into a noted top), never a silently-narrowed
answer; ⊤
```
> *"This message actively sent me the wrong way. It reads as a statement about
> MY declared box… I went back and re-checked my envelope twice."*
> *"It spends 60 words defending its design posture and zero words on provenance."*

**In both external cases the `-inf` was stelling's own ⊤ from upstream**, not
anything the user declared.

**Proposed:**
```
'sqrt' declined: its argument is ⊤ (unbounded), so non-negativity cannot be
established. THE ⊤ DID NOT COME FROM YOUR DECLARATION — it propagated from
equation 12 ('square'), which has no interval transfer. Fix that one first;
this decline is downstream of it.
(If the argument were genuinely negative over your declared box, the message
would name your declaration instead.)
```
**Rule: a decline that reports a box must say where the box came from** — which
equation produced it, and whether it originated in the user's declaration or in
upstream propagation. **Cut the design-posture prose**; it is the ratio both
agents objected to.

> **Applied.** The rule shipped and is what the §1 line above quotes: the
> decline distinguishes *"stelling's own ⊤ from 'div' … (its interval transfer
> declined this form)"* from *"the declared input itself (declared at …),
> spanning [1.0, 2.0]"*, and names the equation and source line for each.
> Pinned by `tests/test_div_straddle_decline.py`.

---

## 3. The unsupported-primitive message

**Today:**
```
escalation declined: primitive 'square' is outside the supported emission set
```
Rated 5/10: *"names the primitive, which is the right half… does not say whether
an interval row also exists, whether this is a policy refusal or an unbuilt row,
or how to add one."*

**Proposed:**
```
escalation declined: 'square' has no emission rule (it also has no interval
transfer, so the interval leg returned ⊤ for it — see the coverage line).
This is an unbuilt row, not a policy refusal.
The supported sets are listed in docs/supported-primitives.md.
```
**And publish the list.** Both agents independently asked for it; there are two
disjoint whitelists and neither is published. *"That table is the highest-value
missing page."*

**2026-08-03 — the specimen closed, the finding did not.** `square` now has an
emission row, so the message quoted above is no longer reachable for it: an
obligation over `jnp.square` reaches the solver. The evaluator's reading is
recorded as it was read and is not restated; what it was *about* — a decline
that names the primitive and nothing else — is still the point, and its live
specimen is now `abs` (`tests/test_unsupported_emission_message.py`, which
asserts its exemplar is outside the emission set before using it). Note also
that the proposed text's parenthesis, *"it also has no interval transfer"*, was
never true of `square`: it had a `sound` interval row the whole time, which is
exactly the contradiction the evaluator could not reconcile.

> **Applied.** Pinned by `tests/test_unsupported_emission_message.py`, whose
> own docstring opens by naming this section and which asserts the shipped
> phrase *"an unbuilt row, not a policy refusal of the form"*. That test
> derives the interval-row fact and its tier from the LIVE registry rather
> than restating it, so the parenthesis that was never true of `square`
> cannot be reintroduced for another primitive.

---

## 4. The element-budget message

**Today** — accurate, quantitative, and **it names something measured
non-binding**:
```
obligation needs 101158 element terms and 2432 root conjuncts, over the
per-obligation emission budget of 512 ... see stelling.obligation.ELEMENT_BUDGET
```
It is *confident and specific*, which makes it **more** likely to be acted on.
A user goes and tunes `ELEMENT_BUDGET` and gets nowhere.

**Proposed:**
```
obligation not attempted: it needs 101158 element terms, over the per-obligation
budget of 512.
The budget bounds what escalation will ATTEMPT; it is not a diagnosis. In the
cases measured here, obligations past the budget declined again on a different
cause once it was raised. Shrinking the obligation is more likely to help than
raising the budget — a tighter interval transfer, a smaller declared array, or
a per-element rather than whole-array property.
```

> **Applied verbatim**, opening clause included. `tests/test_budget_message.py`
> asserts `"obligation not attempted:" in reason`, and the composed sentence is
> at `src/stelling/obligation.py`'s `obligation not attempted: it needs …`.

---

## 5. The relational-`assume` refusal

**Today:** explains why, precisely and honestly, and offers no next step.
Rated *"10/10 as an explanation, 0/10 as a next step"* — and the practical effect
is that **adding a true hypothesis removes the solver you had.**

**Proposed:**
```
escalation declined: this precondition relates two declared inputs (a - b >= 0)
rather than bounding one, so it cannot be emitted as a box, and emitting over
the unconstrained box could produce a witness that violates it.
WHAT WORKS TODAY: a precondition that bounds a SINGLE declared input narrows its
box and does not disable escalation. A relational one needs the predicate emitted
as an SMT assertion, which is not implemented.
NOTE: without the assume this query DOES escalate. If you only need the
unconditional result, removing the precondition restores it.
```
That last line matters: **a user who adds a hypothesis and loses the solver
should be told the trade explicitly.**

> **Applied, with one clause of the draft DROPPED — and the drop is the useful
> part.** `tests/test_constrained_refusal_message.py` pins each shipped
> sentence against the mechanism it describes, and asserts
> `"removing the assume removes it" in CONSTRAINED_ASSUME_REFUSAL`: the trade
> is stated.
>
> The draft's *"WHAT WORKS TODAY: a precondition that bounds a SINGLE declared
> input … does not disable escalation"* was **false in this tree** — a
> constraining assume is exactly what fires this refusal, whether it bounds one
> input or relates two — and it is not in the shipped text. The same test
> asserts `"does not disable escalation" not in CONSTRAINED_ASSUME_REFUSAL`, so
> the false clause cannot come back. What shipped in its place is the form that
> does work: state the bound in the declaration's own envelope, which narrows
> the identical box without a constraining assume. The draft above is left as
> drafted; this note is the delta.
