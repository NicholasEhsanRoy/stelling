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
