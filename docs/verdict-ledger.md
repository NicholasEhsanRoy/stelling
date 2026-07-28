<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Verdict ledger

**What this is for.** When a change to stelling moves a recorded verdict — a corpus
entry, a documented result, a figure in a paper — the verdict moved for a reason,
and the reason has to survive longer than the pull request that caused it.
Otherwise the next person to see the old number has no way to tell a fix from a
regression.

**One entry per verdict that moved.** Four fields, and they are the minimum that
makes an entry useful a year later:

| field | why it is here |
|---|---|
| **what** | which query, precisely enough to re-run |
| **from → to** | the two verdicts, not a description of them |
| **cause** | the change that moved it, in one sentence |
| **commit** | so the diff is one command away |

**Categories.** Borrowed from the release policy, because the distinction that
matters is whether the move needs anyone to act:

- **(a) benign re-baseline** — the tool got a capability it lacked, and a query
  that used to decline now decides. Nothing downstream is invalidated. Record it;
  do not block on it.
- **(b) correction** — a previously *rendered* verdict was wrong. Anything that
  cited it needs revisiting, and the entry says what.
- **(c) regression** — the tool lost the ability to decide something it could
  decide before. Needs an owner.

An entry is cheap. The absence of one is what costs: this ledger exists because
the merge below was the first verdict flip in the project's history and the
checklist that required an entry for it had nowhere to put one.

---

## Entries

### 2026-07-28 — the flagship's `HeatNode` maximum principle

| | |
|---|---|
| **what** | `HeatNode` discrete maximum principle at the refuting configuration — `α = 1.0`, `n_cells = 4`, `dt = 0.1`, `T ∈ [0, 100]^4` in float32, asserting the returned temperature `≤ 100.0`. Reproduce with `flagship_from_main.py` in the sweeps repository. |
| **from → to** | `UNKNOWN` (emission gap: `scatter` outside the supported emission set) → **`REFUTED`**, with a witness confirmed by exact-rational replay |
| **cause** | The static-index `scatter` / `scatter-add` emission rows were registered, so the obligation reaches the solver instead of declining. |
| **commit** | `00333fe` |
| **category** | **(a) benign re-baseline.** The old verdict was a decline, not a claim. Nothing that cited it is invalidated. |

**The witness, because this entry is the one people will check:**

```
x0_0 = 847249408/13421773   x0_1 = 0   x0_2 = 0   x0_3 = 847249408/13421773
T           = [63.125    0.      0.     63.125]
node output = [63.125  101.0   101.0   63.125]
max = 101.0  against the declared bound 100.0  ->  margin +1.0
```

**What did NOT move, and it belongs in the same entry.** The "holds" side of the
same sweep is still `UNKNOWN`. `scatter` remains in
`VERIFIED_BARRED_PRIMITIVES`, so obligations touching it are withheld from
solver-path `VERIFIED` until the class-level audit completes. A reader who sees
only the flip above would otherwise reasonably infer the whole table moved.
