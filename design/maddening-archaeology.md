# The MADDENING archaeology and the evaluate/iterate split — registered before reading

**Status:** REGISTRATION, 2026-07-18. Committed before any MADDENING
history or node source is read (repo identity verified only:
`/home/nick/MSF/msf/MADDENING`, 171 commits). Two arms.

## Arm A — the custom-code archaeology (§2)

**The question:** do the guards in the author's own JAX stack have
receipts — introducing commits that reference failures actually hit?
N=1, on precisely the population (custom solver-shaped code) that no
tracker can see.

**Git only. The author's recollection is not input.** He knows why he
added his guards, he has been inside this investigation for days, and he
knows which answer would be pleasant — the contaminated source every
registration here exists to exclude. The principled reason besides the
suspicious one: the library arm ran on git alone because nobody could ask
Kidger; a weaker standard on the author's own repo would make the two
arms **incomparable**, and comparability is the point.

**Precondition:** history legibility, as in the library arm — plus a
confound that arm didn't have: **MADDENING was built iteratively with AI
coding agents**, so a guard added inside a large agent-authored sprint
commit may carry no failure narrative in its message even when there was
one. The commit-size distribution is checked and reported first.

**`comment-grade` is hereby activated** (banked recorded-not-used in the
drift probe; this is the registration it was banked for): a failure
narrative present in a source comment but absent from the commit message
is its own bucket, reported separately, and **it does not upgrade to a
receipt** — the library arm refused that upgrade (equinox DAZ) and this
arm gets the same rule.

**Corpus:** MADDENING's guards, enumerated by grep **before any digging**
— `where`-clamps, safe-division/sqrt patterns, NaN/finiteness checks,
capacity/overflow flags, epsilon guards — count reported, then
`git log -S` per site.

**Buckets** as in the library arm: **receipt** (added later + commit or
linked issue describes the failure) / **ambiguous-hardening** (added
later, no story) / **comment-grade** (narrative in source only) /
**foresight-shaped** (present in the line's first version).

**Bands — fixed:**

| finding | reading |
|---|---|
| ≥2 receipts | custom-solver authors pay, with receipts — N=1, the only sample that exists |
| 1 receipt | weak; report the site, not a rate |
| 0 receipts, history legible | **ambiguous, not falsified** — foresight is consistent with *easy* and with *paid elsewhere* |
| history illegible | no result; say so |

**The comparison rule:** the library arm found 3 receipts across the
examined sites in 2 libraries; MADDENING's rate is reported against that
**with the sample sizes in the same sentence** — both are tiny; the
comparison is a gesture, not a statistic. And plainly: **N=1, the
author's own code, the author wants a particular answer — the weakest
sample this project has drawn, and the only one available; the
alternative is an argument.**

## Arm B — the evaluate/iterate split (§3)

**The prediction (from the supply probe):** a node that *evaluates*
(feed-forward, explicit stencil, forward model) is vector-field-shaped —
proves in ~2 rounds, box-shaped, discharges no incident. A node that
*iterates* (implicit solve, contact resolution, adaptive stepping,
acceptance criterion) is solver-shaped — the live case the 17-of-20
finding is about.

**Classification rule, fixed (the rule matters more than the counts —
it must transfer to anyone's code):** a step function is **solver-shaped**
iff it contains an inner iteration whose count or continuation is
**data-dependent** (a convergence test, an acceptance test, a fixed-point
or root-finding loop, adaptive sub-stepping); it is **evaluate-shaped**
iff it is a fixed dataflow executed once per external step, branches
included.

Scope: **all nodes** (the falsifier binds to nodes, as ordered) — and the
core scheduling/coupling machinery is *also* inventoried and reported,
separately labeled, so solver-shaped-ness in the framework cannot be used
to dodge the node-level falsifier, nor be missed by it.

**The falsifier:** if every MADDENING node is vector-field-shaped, then on
this evidence **the tool would not help the author's own work** — reported
as loudly as the alternative.

**Optional follow-up, budget permitting:** the supply-probe method (pen +
Z3, stelling forbidden, four-hour analogue) on one solver-shaped node if
one exists. Not barred by the self-containment rule: that rule exists
because a *bug-finding* null on one's own code is uninterpretable —
**provability is not contaminated by authorship**. The classification is
the deliverable; the follow-up is not.

---

# Reading (2026-07-18 — git only, as registered)

## Precondition

Root is organic (2 files, 663 insertions, 2025-02-27 — not a dump). The
middle is **sprint-shaped**, as the registered confound predicted: an
88-file "MADDENING v1: Complete modular simulation framework" commit and a
175-file compliance-infrastructure commit anchor the distribution. Rule
applied per-site: guards born inside sprint commits read as
foresight-shaped and may hide narratives — the comment-grade bucket is why
this arm can still see them.

## Arm A — the digs (169 guard-pattern hits enumerated across ~14 files)

| site | introduced | bucket |
|---|---|---|
| **`cdd.py` high-contrast regression guard + `MAX_OUTER=200`** | `c0953ad` 2026-07-16, modifying an existing file; message names the guard, the in-diff comments narrate the failure mode ("*unsafe at high coefficient contrast… the budget fills*"), and a 100-line regression test lands with it | **RECEIPT** (message+diff grade — the same grade as diffrax's "Crash fixes") |
| the May CFL episode | docs-only: "FVM failures resolved — **CFL not dtype was the load-bearing fix**" + "adopt peer agent's **float32 diagnosis**" (2026-05-19) — a failure hit, first misdiagnosed, resolved by parameter change; **no code guard added** | **comment-grade** (narrative in docs, not a defence change) |
| heat.py `MADD-ANO-002` metadata ("CFL stability **not enforced at runtime** — unstable timesteps silently produce incorrect results") | `7ee6cff` 2026-03-12 — the compliance sprint, **two months before the failure it describes was hit** | foresight-shaped |
| `safe_denom` (Aitken accelerator) | with its feature (2026-03-14); site comment describes mechanism, not incident | foresight-shaped |
| `safe_error` | inside the v1 sprint dump | foresight-shaped (sprint caveat) |
| `qn_finite` isfinite gate (IQN-IMVJ) | with its feature (2026-05-30) | foresight-shaped |
| lbm/lbm_pipe wall masks; `health_check` node | inside the 175-file compliance sprint | foresight-shaped (sprint caveat) |

**Band: 1 receipt → weak; the site is reported, not a rate.** Comparison,
with sample sizes in the same sentence: the library arm found 3 receipts
across two libraries' examined defence classes; this arm found 1 receipt
in 1 repo — both samples are tiny and the comparison is a gesture, not a
statistic. And plainly, as registered: N=1, the author's own code, the
author wanted a particular answer — the weakest sample drawn, and the
only one available.

**The observation the buckets don't capture (recorded, not counted):** the
CFL arc — *documented as an unenforced obligation in March, hit in May,
misdiagnosed as dtype, fixed by parameter, still unenforced today* — is a
named anomaly (`MADD-ANO-002`) that cost a debugging episode **after**
being written down. Documentation-as-defence did not defend. The
obligation is the founding document's own Stage-0 demo property (CFL ⇒
stability), sitting unenforced in the author's own heat node.

## Arm B — the evaluate/iterate classification

Rule as registered (data-dependent inner iteration = solver-shaped):

| class | nodes | grounds |
|---|---|---|
| **solver-shaped** | **the adaptive wavelet family**: `WaveletEllipticNode` + its stack (`cdd.py` — `lax.while_loop`, `MAX_OUTER=200`, Dörfler budget; `wavelet.py`, `hierarchical_hat.py`, `topk.py`, `matrixfree.py`) | data-dependent outer iteration with a convergence test and an iteration cap |
| **evaluate-shaped** | ball, table, heat, spring, rigid_body, rigid_body_2d, lbm, lbm_pipe, heart_pump, health_check | fixed dataflow per external step; no data-dependent continuation |

**Core machinery (separately labeled, as registered):** heavily
solver-shaped — the coupling fixed-point loop (`group.py`,
`lax.while_loop`), the Aitken/IQN-IMVJ accelerators (`acceleration.py`),
the IFT implicit solver (`implicit.py`), `graph_manager`'s loop, the
multi-GPU `iterative_solver.py`, and `simulation/adaptive.py`.

**The falsifier does not fire** — a genuinely solver-shaped node exists —
and the majority finding is reported at equal volume: **most nodes are
vector-field-shaped**, the supply probe's cheap-and-useless case; the
solver-shaped surface is one node family plus the framework core.

**The coincidence that matters:** the one receipt sits **on the
solver-shaped node** — the iteration-budget guard of the wavelet solver.
The author pays where the code iterates, on an N=1 sample, which is the
work order's thesis in one data point and no more than that.

**Follow-up not run** (budget; the classification is the deliverable). The
natural target is recorded: `cdd.py`'s outer loop — an iteration-cap
termination obligation identical in shape to diffrax#386's, on the
author's own code.

## Inversion (2026-07-18): the core is the finding, not the nodes

The reading above filed the core as "separately labeled." Inverted: the
coupling fixed-point loop, the Aitken/IQN accelerators, the IFT implicit
solver, and the multi-GPU iterative solver **are MADDENING's library
internals, and the author is their maintainer** — the 17-of-20 customer
(*a maintainer verifying control loops*) instantiated in his own repo.
Every node, evaluate-shaped or not, inherits the core: the
maintainer-inheritance argument arriving at home instead of in diffrax.
The natural supply-probe target is corrected accordingly: **the core's
coupling fixed-point loop**, not `cdd.py` — its termination is the same
obligation shape as diffrax#386's, and it is the thing every node depends
on.
