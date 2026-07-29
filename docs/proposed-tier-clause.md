<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# A gauge clause for `TIER_EXACT` — PROPOSED, and it runs the OPPOSITE way round

`TIER_EXACT` says the returned box **is the hull of the achievable image**, not
merely a sound enclosure of it. The tier rides into the verdict stamp. It was
asserted in prose for every row and checked by nothing, and a blinded audit
found `sign` claiming it falsely — by *reading* the claim, not by running
anything.

## The correction that makes this buildable

The first prototype sampled the box on a `linspace` grid and refuted `exact`
when the box was wider than the sampled image. **It refuted `abs([-3, 3])`,
which is genuinely exact**, because the grid never landed on 0.

That is not a tuning problem. It is the wrong direction of inference, and the
right one is its converse:

> Let `S` be a set of sampled image points, `H(S)` its hull, `I` the true image
> hull, and `B` the returned box. Soundness gives `H(S) ⊆ I ⊆ B`.
>
> - **If `H(S) = B`, then `H(S) = I = B`. The row is EXACT on that box, proven.**
> - **If `H(S) ⊊ B`, nothing follows.** The samples may simply have missed an
>   extremum.

So a **PASS is a proof and a FAIL is not a refutation** — the reverse of what
the prototype claimed and of the framing that commissioned it. A clause built on
the refuting direction produces false alarms; one built on the confirming
direction cannot.

## Seeding, which is the same norm one level up

`H(S) = B` is only reachable if `S` contains the image's extremes. For a
piecewise-monotone function those sit at the box endpoints or at a **breakpoint**
— a value where the transfer's behaviour changes. The transfer already knows
them; they are its `elif` boundaries. So the clause asks each row to declare
them, and seeds from that rather than from a grid:

| row | breakpoints | why |
|---|---|---|
| `abs` | `0` | the sign change |
| `square`, `integer_pow` | `0` | the even-power minimum |
| `neg`, `sqrt`, `exp` | none | monotone throughout |
| `sign` | `0`, `±MIN_NORMAL` | the three definite branches |

**This is the session's own norm applied to the checker**: build the enumeration
from where the behaviour changes, not from where the author happened to sample.

## Measured, on the rows in tree

```
row     box                   transfer          seeded image hull   tier    verdict
abs     [-3, 3]               [0, 3]            [0, 3]              exact   PASS — proves exact
abs     [1, 2]                [1, 2]            [1, 2]              exact   PASS — proves exact
abs     [-2, 0]               [-0, 2]           [0, 2]              exact   PASS — proves exact
neg     [-3, 3]               [-3, 3]           [-3, 3]             exact   PASS — proves exact
square  [-2, 3]               [0, 9]            [0, 9]              sound   PASS — proves exact
sqrt    [1, 4]                [1, 2]            [1, 2]              sound   no proof   (outward rounding)
exp     [0, 1]                [1, 2.71828]      [1, 2.71828]        sound-libm no proof (1-ulp bracket)
sign    [-3, 3]               [-1, 1]           [-1, 1]             sound   PASS — proves exact
sign    [5e-324, 1e-320]      [0, 1]            [0, 0]              sound   no proof
```

Three things worth reading off it:

- **`abs` passes now**, at the box the unseeded prototype refuted. The seed `0`
  is the whole difference.
- **`sqrt` and `exp` correctly fail to prove exactness** — their boxes are
  outward-rounded, so the check discriminates rather than passing everything.
- **`sign` at the subnormal band fails to prove it, and that agrees with its
  registered tier.** The box is `[0, 1]`; under flush the image is `{0}`. The
  row is registered `sound` for exactly that reason, so the clause and the
  registry agree — which is the check working.
- **`square` proves exact on THIS box while registered `sound` — and that is
  the scope limit biting its own example, not an upgrade candidate.** The
  clause proved *per-box* exactness on a box where the arithmetic happens to be
  clean, and reading it as a tier claim is exactly the misuse the scope section
  warns about. Measured, on the same row:

  ```
  square [-2, 3]    transfer [0.0, 9.0]     image hull [0.0, 9.0]      PASS
  square [-0.1, 0.3] transfer [0.0, 0.09]   image hull [0.0, 0.09]     PASS
  square [0.1, 0.3]  transfer [0.01, 0.09]  image hull [0.010000000000000002, 0.09]
                                                                       no proof
  ```

  **`square` is correctly registered `sound`.** The distinguishing factor is
  not whether the squares are representable — `0.09` is reproduced identically
  by both routes. It is that a **straddling** box takes its lower endpoint from
  the even-power rule (exactly `0`, no arithmetic) and its upper from a single
  squaring, while a **non-straddling** box computes both by squaring, and
  outward rounding then shows up — here on the LOWER endpoint, `0.01` against
  an executed `0.010000000000000002`. A clause that reported this as an upgrade
  candidate would be inviting someone to act on a per-box proof as though it
  quantified over every box.

## The proposal

A gauge clause asserting, for every row registered `TIER_EXACT`, that
`H(S) = B` at each seeded box. Failure means **either** the row is not exact
**or** the seed set misses an extremum — and the message must say both, because
the second is a test-authoring bug with a local fix and the first is a
soundness-adjacent defect.

## Scope, stated because the clause cannot cover the claim

- **Per-box proof; the universal claim stays untested.** `TIER_EXACT` quantifies
  over every box. This proves it for the boxes tested. It is a battery, like
  every other gauge here, and it should not be described as verifying the tier.
- **Unary piecewise-monotone rows only.** For a binary row the extremes of a
  monotone function sit at the corners, but `rem` is monotone in neither
  argument — sawtooth in the dividend — so corner evaluation does not find them.
  `rem` is registered `sound` and this clause has nothing to say about it.
- **It cannot check `sound` or `sound-libm`.** Those claim enclosure, which
  sampling can only fail to contradict.
- **Breakpoint declarations are themselves unverified.** A missing breakpoint
  produces a false FAIL, never a false PASS — so the failure direction is
  toward noise at authoring time, not toward certifying a wrong tier.

**If the clause were built on the refuting direction instead, it should not be
built at all**: it would send a reader chasing F4-shaped ghosts, and the `abs`
result above is what that looks like.
