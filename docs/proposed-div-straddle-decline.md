<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Real-mode `div` should DECLINE on a straddling divisor — PROPOSED, NOT APPLIED

Published surface, so this is the argument.

## The finding: proposal #1's "worst message" has a cause, and it is a missing decline

`docs/proposed-decline-messages.md` opens with the message two independently
blinded external agents rated **1/10** — *"This message told me nothing"*, *"this
is where I'd have quit"*:

```
assert #0: unknown — undecided for 2/2 element(s)
```

Measured, real mode, a divisor declared over `(-1.0, 1.0)`:

```
obligation : unknown          coverage : known=5/5 unknown=0
notes      : (NONE)           detail   : 'undecided for 1/1 element(s)'
```

**That is the same message, and here is its cause.** `iv.div` returns
`[-inf, +inf]` for a zero-straddling divisor. That is a sound box, and it is not
a decline: nothing is raised, nothing is noted, and **coverage reads 100%
known** while the answer carries no information at all.

## Why this matters more than the message

The same document proposes propagating the `div` guard's message, rated **9/10**
by one agent and used as the model for the other:

```
escalation declined: 'div': divisor may be zero over the declared box — SMT-LIB2
division is underspecified at 0 (element 0 spans [-69.4, 69.4])
```

**That message is on the ESCALATION face.** A query that discharges or fails on
intervals never reaches it. So the tool's best message and its worst are the
same situation on two different paths, and **the default path gets the worst
one** — which neither external agent could report, because both reached
escalation.

**A decline has to exist before a message can attach to it.** That ordering is
why this proposal comes before #1, not after.

## The general shape, measured

Across the 46 registered transfers **as the registry stood when this table was
measured** — the index-bounds round has since taken it to 48, adding
`dynamic_slice` and `dynamic_update_slice`, and the partition below is not
restated for them; `docs/supported-primitives.md` is generated from the live
registries and is the current count:

| decline shape | count |
|---|---|
| silent ⊤ only (`return None`, no reason) | 8 |
| loud only (raises with a reason) | 5 |
| both paths | 33 |

**The silent-⊤ class is invisible to every instrument that counts declines**, and
`div`'s straddling case is worse than any of the eight, because it does not even
return `None` — it returns a *box*, so it is not a ⊤ in the coverage census
either. It is a decline wearing an answer's clothes.

## The proposal

Real-mode `_t_div` raises when the divisor's interval contains zero, with the
interval printed — the three properties the 9/10 message has (names the
primitive, gives the reason, prints the box). Integers are unaffected: they route
through the overflow guard, which already refuses.

## What it costs, measured by the counterfactual method

Stubbed in-process, destroyed after, registry asserted restored:

```
baseline                     : 1275 passed, 2 skipped
with the straddling decline  : 1275 passed, 2 skipped
```

**Zero.** No test depends on `div` returning a wide box rather than declining,
which is the expected shape: the wide box already poisons everything downstream,
so the verdicts were already `unknown`. **What changes is the bookkeeping** — the
coverage number stops reporting 100% known for a query that knows nothing, and
the cause becomes attributable.

Per Norm I the stub is *more restrictive* than the change would be, so a zero
here is the conclusive direction: a real implementation can only break less.

## What this does NOT claim

- **It does not fix any verdict.** Nothing moves from `unknown` to a verdict;
  the value is entirely in the diagnosis and the honest coverage number.
- **It does not address the other EIGHT silent-⊤ rows** (`convert_element_type`,
  `copy`, `dot_general`, `gather`, `reshape`, `scatter`, `split`, `unstack`).
  Those return `None`, so they *are* counted as ⊤ — they lack a reason, not a
  decline. Different fix, smaller.

  *This line said "seven" while listing eight names, and the error propagated
  into a later session's brief. The verified count is the list itself: eight
  rows. A finer partition of the 46-row registry (silent-only / loud-only /
  both / cannot-decline) was corrected here as "8, 5, 13, 20" from a script
  that was not retained, and it does not re-derive: a source census at this
  proposal's revision gives 7 / 11 / 14 / 14
  (stelling-sweeps/verify_9b555_replacements.py), and an execution probe
  counted 13 rows with a reachable no-reason ⊤, 3 of them silent-only under
  strict reachability. Three instruments, three partitions — the partition is
  operationalization-dependent, so none is quoted as THE count. The earlier
  "33 both" conflated two buckets of one partition. All of these are the
  count-error class recorded in CONTRIBUTING.md, made inside documents about
  message quality.*
- **It has not been measured on external code.** Both external agents reached
  escalation; whether a first-time user on the default path hits this is
  unmeasured, and re-running them against the fixed tool is the measurement.
