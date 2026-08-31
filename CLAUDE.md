<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Working in this repository

`stelling` is an assertion-based verifier for JAX. It emits VERIFIED,
REFUTED, UNKNOWN and DECLINED stamps that people act on, so **a stamp that
claims more than the analysis established is a defect of the same kind as a
wrong answer** — not a documentation nit.

This file is pointers. It carries no counts and cites no test names, because
anything restated here would go stale and become the thing it warns about.

## Read these before changing anything

| document | what it is |
| --- | --- |
| `design/lessons-ledger.md` | **Cross-pass principles, durable and reviewed.** Read it first. Each entry is a defect class this repository has actually shipped, with the measurement that found it. Most new work rediscovers one of them. |
| `SOUNDNESS.md` | The soundness log, its response policy, and what each released version is known to get wrong. |
| `ARCHITECTURE.md` | The layers, and which of them are core-audited. |
| `DOCUMENTATION_ARCHITECTURE.md` | Where a given kind of claim is allowed to live. |
| `CONTRIBUTING.md` | Environment, extras, and how to run the suites. |

`design/lessons-ledger.md` is the one to actually read rather than skim. Two
of its entries carry most of the weight:

* **A check that MODELS a behaviour is always one indirection behind it.**
  Assert on what the program did, not on source text that describes it. A
  scan over spellings loses to the next spelling, permanently.
* **A control drawn from the space you are already reasoning in tests your
  model, not the world.** A control that fires proves the instrument can see
  your own writing; it proves nothing about the corpus.

## The habits that this codebase is built on

Not style preferences — each one is here because its absence shipped a defect.

* **Derive counts and sets; do not type them.** A number written beside the
  thing it counts goes stale silently. If you must state one, state what
  measured it and when — and ask whether a fresh process would read the same.
* **Bound every set from both sides.** Shrink it and watch something redden;
  grow it and watch something redden. Then bound its *consumption* too: a
  constant with the right members, consumed through a slice, is an unbounded
  control wearing a guard.
* **Bound what the system produces against what it consumes.** If a component
  writes several witnesses to one fact and the decision reads some of them,
  the next defect lives in the ones nothing reads.
* **Everything upstream of your assertion is an input, and an input you did
  not check is an input you assumed.** Before reasoning *from* a value, ask
  whether the thing that writes it could have written it — and derive that
  invariant from the writer, not from the reader's expectations. Guards here
  have repeatedly been correct about a class and still wrong about which
  *rung* the class sits on: source, then the object the source builds, then a
  tally of what ran, then the outcome actually measured. Each rung reads like
  a fix until the next attack arrives.
* **Say what the search found, not what exists.** *"The sweep finds no input
  that tells it apart"* is a measurement. *"No input tells it apart"* is a
  claim nobody built a control for.
* **A disclosed limit beats a closed one you cannot justify** — provided the
  disclosure is the true class. Narrowing a limit around the attack that was
  just demonstrated is how a disclosure becomes false.
* **Green suites are not a finding.** A central claim can be false while
  everything passes, because no test bounded the thing that was wrong.

## Process documents, when a review campaign is running

Binding process for builder and auditor agents lives outside this repository,
in a sibling `stelling-sweeps/process/` directory — house rules, the audit
protocol, and the component specs. They live outside the tree because they
change faster than a release does, and they are not part of the published
artefact. **If that directory is present, read `HOUSE-RULES.md` and
`AUDIT-PROTOCOL.md` before starting**; the audit protocol's Rule R governs how
to answer a driven falsification, and following it is not optional.

If it is not present, the ledger and the habits above still apply, and nothing
below concerns you.

### The build–audit loop

Soundness-adjacent work here lands through a loop, not a review. A **builder**
implements against a written spec; an independent **auditor** then attacks the
result and returns one of exactly three verdicts — `MERGE`, `MERGE WITH FIXES`
(naming them precisely enough that someone else can apply them), or
`DO NOT MERGE` (a defect it can *drive*, not one it can argue for). Anything
short of `MERGE` starts another round. Components routinely take several.

Three properties make it work, and each is load-bearing:

* **The auditor gets THE TREE, never the builder's transcript.** An author
  cannot be their own auditor, and an auditor who reads the author's reasoning
  inherits the author's blind spot. If you are auditing, do not go looking for
  how the code came to be — measure what is there.
* **A verdict with no driven attack behind it is not a verdict.** *"I read it
  and it looks right"* is `DO NOT MERGE` with the reason "not audited".
  Neutralise each refusal in turn and re-run: a guard whose removal reddens
  nothing is a guard nothing holds.
* **Builders may — and do — correct the spec and the audit.** Several
  specification errors in this project were found by builders driving a claim
  the spec asserted. If you conclude a mandate is wrong, say so with the
  measurement that shows it and stop, rather than building to it.

If you are asked to build or audit, you are one of these roles and not both.
Ordinary work in this repository is not a round of the loop and should not
pretend to be one.

## Two things that are always true here

* **Never weaken a marker to get green.** A strict `xfail` that starts
  XPASSing is a repair announcing itself, and the disclosure is meant to move
  with the fix. Silencing it discards the signal the marker exists to send.
* **Report outcomes faithfully.** If a run was red, say so with the output; if
  a step was skipped, say that. A number with no driver behind it is a
  sentence, not a measurement.
