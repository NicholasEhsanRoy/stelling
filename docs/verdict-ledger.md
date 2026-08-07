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
`VERIFIED_BARRED_PRIMITIVES`, so an obligation whose EMITTED SLICE carries it is
withheld from solver-path `VERIFIED` until the class-level audit completes. A
reader who sees only the flip above would otherwise reasonably infer the whole
table moved.

Two things the bar's scope does **not** cover, both easy to over-read:

- The membership is **exact-name**, so `scatter-add` — what `segment_sum` and
  `x.at[idx].add()` lower to — is **not** barred. `design/scatter-rows.md`
  records a completed fresh-adversarial-auditor pass over the accumulate rows.
- The scope is the **decided obligation's slice**, not the traced query. A
  `scatter` elsewhere in the jaxpr, on an obligation interval arithmetic
  settled, withholds nothing: the emission row was never asked about it. The
  bar read the whole query until it was slice-scoped, and did withhold those,
  so a ledger entry recorded before that may name a withheld `VERIFIED` this
  build renders. The narrowing is logged as a soundness event in
  `SOUNDNESS.md` — it moves verdicts, in the UNKNOWN → VERIFIED direction, and
  a reader comparing an older entry to a fresh run needs that entry to know
  which of the two builds to trust.
- The scope is **derived from the query**, not read off the escalation: the
  bar re-slices the decided obligations out of the `ClosedJaxpr` it was handed,
  and no record can assert that its own slice was clean.
- Narrowing to that slice is **earned per obligation**, by reproducing BOTH
  the script hash and the slice fingerprint the recorded invocation carries.
  Deriving the scope alone did not bind the escalation to the query —
  nothing did, until the pairing gate below — and neither did the script
  hash: the barred row emits no text, so a
  scatter-bearing slice can emit byte for byte what a scatter-free one emits
  (measured, `SOUNDNESS.md`). Both mispairings cleared the bar until the
  fingerprint was added. What that buys is bounded and stated there: a
  mispaired escalation gets the whole-query bar, but the hashes are carried by
  the record, so this is a defence against an accidentally mispaired assembly
  and not against a fabricated one — a caller who can fabricate a record can
  fabricate the verdict.
- **The pairing is a separate gate, and scoping the bar did not cost it — it
  revealed it was never there.** An earlier version of this bullet said
  scoping "cost a backstop": that where two queries' decided slices are the
  same expression, the bar narrows correctly and a mispaired assembly can
  still issue VERIFIED on a query that is REFUTED, and the whole-query bar had
  withheld that by accident. The second half is what was wrong. That accident
  only ever covered queries carrying a barred primitive — the only ones any
  version of the bar looks at — and the identical false VERIFIED is reachable
  on a query with **no scatter anywhere**, on every build. So the finding is
  not a cost of scoping; it is that `make_solver_verdict` never bound its
  three arguments to one query. It now binds two of them: `escalate` records
  the query's content hash and assembly refuses an escalation produced on a
  different one (`MispairedEscalationError`). `propagation` carries no such
  hash and is not bound; what that leaves open is stated in
  `make_solver_verdict`'s docstring and measured in the suite. `SOUNDNESS.md`
  logs both.
