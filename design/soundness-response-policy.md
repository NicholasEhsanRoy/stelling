<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Responding to a soundness defect

**Written 2026-08-28, on the first occasion it was needed.** The occasion is
recorded at the end, because a policy written from one case should say which
case, and a reader deciding a future one is entitled to see what the first one
looked like.

`stelling` exists to say VERIFIED and be right. **A defect that can mint a
false VERIFIED or a false REFUTED is not a bug of the same kind as anything
else in this repository**, and it does not get the same response. This file
says what the response is, so that it is a decision taken in advance rather
than one taken while looking at a specific unpleasant fact.

## What counts

A **soundness defect** is anything that can make the tool assert something
false about a program: a VERIFIED the program contradicts at a point of the
declared box, a REFUTED that is true there, or a verdict whose stamp claims
more than the analysis established. **Vacuity is included**: a VERIFIED over an
empty set is a false claim in the only sense that matters, because a reader
acts on it.

Not soundness defects, and not covered here: precision losses (the tool
withholding more than it could), instrument gaps that cannot themselves produce
a wrong verdict, and documentation that is merely incomplete rather than false.

## The four questions, in order

Answer these before choosing a response. Each is a measurement, not a
judgement, and each must be **driven** rather than argued.

1. **Can it mint a false verdict?** If not, this file does not apply.
2. **Is it reachable from an ordinary program**, or only from a construction
   nobody writes? Reach is measured by building the program, not by reasoning
   about it. *"I could not find one"* is a reading only if the search could
   have spoken.
3. **Is it in a published artefact?** Check every published version, not the
   current tree — a defect can predate the code you are looking at. This is a
   `pip download` and a re-run, and it is cheap.
4. **Could anyone have acted on it?** Download counts, traffic, known users,
   any citation of a verdict. A number that is bot-shaped is not a user.

## The ladder

Each rung includes every rung above it. Choose the lowest one that fits, and
**record which rung and why** — the reasoning is the part a later reader needs.

**R0 — Record.** Always, without exception, on discovery and before repair.
A `SOUNDNESS.md` entry naming the defect, its reach, the versions affected,
and what is not yet established. **An unrecorded soundness defect is a second
defect.**

**R1 — Fix forward.** The defect is in the working tree only, or in a
published version nothing can reach it from. Repair, audit, release.

**R2 — Warn on the affected versions.** The defect is published and reachable.
`README.md` and `CHANGELOG.md` say which versions carry it, what it can make
the tool assert, and what a reader should do. Prose reaches someone who
already installed; a yank does not.

**R3 — Yank the affected releases.** The defect is published, reachable, and
mints false verdicts on programs a user would plausibly write. **Yank, do not
delete.** A yank stops new resolution and leaves pinned environments working,
which is the correct asymmetry: it protects the people who have not yet
installed without breaking the people who have. **Yanking is not an admission
of embarrassment; it is the cheapest correct action available**, and hesitating
over it because a release is recent is how unsound artefacts stay resolvable.

**R4 — Notify.** Anyone who could have acted on a false verdict is told
directly. Requires knowing who they are; if question 4 says nobody, this rung
is skipped **and the skip is recorded with its evidence**, so that a later
reader can see it was decided rather than forgotten.

**R5 — Freeze the release train.** No new release until the class is closed.
Reserved for a defect whose *class* is open — where the specific instance is
understood but the general question is not, so a new release would ship the
same unsoundness by another route.

## Two rules that are not rungs

**Yank on verification, not on suspicion, and not later than verification.**
The trigger is a driven witness plus a confirmed reach into a published
artefact. Once both exist the yank is not a separate decision to be scheduled;
it is the same decision.

**A repair does not retire the record.** The `SOUNDNESS.md` entry stands after
the fix, with the fix recorded beside it. The versions that carried the defect
carried it permanently, and someone reading an old verdict needs to know.

## The occasion this was written for

**2026-08-28.** Real mode judges obligations by interval arithmetic over ℝ with
outward rounding and does not model IEEE flush-to-zero, the program's own
float format, or the compiler's freedom to reassociate a reduction. Seven
programs were driven where a VERIFIED contradicts the compiled program at a
point of the declared box — including one whose declared values are all
ordinary normal float64s and whose computed box is `[1.25, 39.0]` while the
program returns `−8.0`.

Answers to the four questions: **(1)** yes, false VERIFIED and false REFUTED,
both driven. **(2)** yes — `x**1001` over `x ∈ [-0.4,-0.2]`, seven repeated
squarings in float32, and a single float32 multiplication. **(3)** yes, and
worse than first thought: four of the seven reach **`v0.1.0`**, the first
published release, so the class is as old as the project. **(4)** no — 260 PyPI
downloads with a release-day spike and a 2–7/day trickle, which is
automation-shaped, and GitHub traffic agrees.

**Rung chosen: R3, plus R5.** Yank `0.1.0` and `0.2.0`; warn in `README.md` and
`CHANGELOG.md`; freeze the release train until the class is closed rather than
the instance. **R4 skipped on the evidence in (4)**, recorded here rather than
omitted.

The reason R5 applies and R3 alone did not: at the time of the decision the
*instance* was understood and the *class* was not. The repair under
consideration had just been refuted — a magnitude-band check that a reassociated
reduction walks straight through — so a release cut on the strength of it would
have shipped the same unsoundness by a route nobody had measured.

**And the honest part.** The instrument that would have found all seven was
considered and deliberately scoped away, in `tests/property/README.md`:
*"An oracle pointed at floats measures the documentation."* That sentence is
the project's own ℝ posture turned into a reason not to look. **A disclosure
that also functions as a reason not to test for the thing disclosed is the
shape to watch for**, and it is why question 2 above insists that a null is a
reading only if the search could have spoken.

## Addendum, 2026-08-30: what the figures settled at

**The section above is left exactly as it was written on 2026-08-28, and
that is deliberate.** It records a decision taken with the evidence
available when it was taken, and a policy whose worked example has been
quietly edited to agree with later measurements is a policy that cannot
show anyone how a decision under uncertainty actually goes. What follows
is what the numbers became, dated separately.

**Seven became nine.** The instrument built to pin the seven found two more
by itself — `subnormal-comparison`, from an unbiased draw while its
classifier was being written, and `assume-narrows-past-the-program`, which
an independent audit drove. Both are registered in
`tests/property/_float_oracle.py`'s `MEMBERS` rather than left in a report,
because **a member found and not pinned is a member that gets lost.**

**Five falsify a discharge and one contradicts a refutation.** The other
three violate box containment without a VERIFIED resting on them.

**"Four of the seven reach `v0.1.0`" became five of nine, and the figure
moved because the population did, not because the measurement did.** Nine
of the ten driven programs violate containment against `v0.1.0`'s own
`src/`; five falsify a discharge there. The one that does NOT reach
`v0.1.0` is `reassociation-n33`, and the reason is worth keeping: at
`v0.1.0` `interval.mul` was not exact, so the pinned product boxed to a
width instead of a point and the reduction's box was wide enough to contain
the executed value. **A later tightening is what opened that member.** A
soundness defect can be introduced by making an analysis more precise, and
this is the project's own instance of it.

**The response did not change.** R3 + R5, R4 skipped on the evidence in
(4). `0.1.0` and `0.2.0` were yanked from PyPI on 2026-08-30 with reasons
naming the FTZ class; the release train is frozen; and question (1) was
already answered before any of the figures above moved. **That is the point
of answering the four questions in order:** the rung was decided by the
first three, and none of the later arithmetic touches them.
