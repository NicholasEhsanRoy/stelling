# stelling Documentation Architecture

**Status:** planning document, v0.2, July 2026. Normative for the artifacts it
names and the discipline around them. On anything technical, `SOUNDNESS.md`
(policy), `design/founding.md` (commitments and roadmap), and the notes in
`design/` outrank it. Like the founding document it is subordinate to evidence:
where contact with real code contradicts a judgement made here, the evidence
wins and this document changes. It has already, twice — see below.

**THE PRESENT TENSE IN THIS DOCUMENT IS JULY 2026'S, AND THE REPOSITORY HAS
MOVED.** This is a planning document and its sentences about the state of the
tree describe the tree it was written against. Three of them said in so many
words that *"the repository is pre-Stage-0"* and that harness primitives, the
query object and the z3 encoder do **not** exist — in the executive summary, in
§2.3 and in Appendix B, in a file that ships in the sdist. All three exist.
`v0.1.0` is tagged, `stelling.__version__` is `0.2.0.dev0`, and the suite is
thousands of tests — run `pytest --collect-only -q` for the figure; it moves
with every commit and is deliberately not written down here, which is
`docs/norms.md`'s rule about figures and the same reason
`.github/workflows/ci.yml` deleted the pass counts that used to stand in it.
`git blame` puts all three sentences at `22c176f`,
2026-07-17; the file was edited on 2026-07-28, 2026-08-20 and 2026-08-21 and
none of the three was touched. They are dated in place rather than deleted,
because the argument each one introduces is about a decision taken at that
moment and does not depend on the moment lasting. **Treat every unqualified
present-tense claim here about what the repository HAS as being as of that
date, and check it against the tree**; `SOUNDNESS.md` is the ledger for
anything that turns on current behaviour. Nothing in this repository gates this
document's tense, which is why three sentences survived three later edits.

**Evidence in this document.** Claims about jax were **verified against jax 0.10.2
and 0.11.0 on 2026-07-16** by probing the installed package; re-verify on every
series bump, per `TESTED_JAX_SERIES` and the discipline in
`design/transparent-primitives.md`. Claims about the FDA computational-modelling
guidance (§13.2) were verified against the final PDF, read 2026-07-16. Claims
about IEC 62304 (§13.1) are **paywalled and reported, not verified** — §13.1 says
so at the point of use and does not assert clause titles. Claims about IEC 61508,
ISO 26262, DO-178C/DO-330 and EN 50128 (§12) are **cited from working knowledge
and unverified** — §12.6 is the standing disclaimer and it is not decorative.

**Changes in v0.2.** Four corrections, one addition, one companion document.
Link 7 was mischaracterised as the bet Kani makes on rustc's backend; it is not,
and the correction runs through §2.3, §4.2, §4.3, and §11.1 (**a precision field
in the stamp**). Fuzz-on-verified was under-scoped at links 3–6; its oracle is the
compiled program, so it reaches **2–8** — and *reaches* is the whole of it.
***Corrected 2026-08-24:*** that clause continued *"and it is the only defence
links 2 and 7 will ever have before Phase 5"*, which §2.5's own capability table
forbids two screens later — **the probe is a search, and absence of evidence is
not evidence**. It is the only mechanism this architecture ever *points at*
links 2 and 7 before Phase 5, which is a weaker claim by a category: it leaves
those links with no defence at all, only a chance of being caught. §2.3.1
re-derives the whole column against the tree. The chain stopped at 7 and now stops at
**8 — hardware**, which is unclosable and named for that reason (§2.3). §13 is new
and asks what §12 does not. `design/value-model.md` is new and is the half of the
decision this document was missing.

**Scope note.** This document names no application sector. stelling is a
general-purpose verifier for JAX array programs; the qualification frameworks it
maps to — IEC 61508, ISO 26262, DO-178C/DO-330, EN 50128 — cover most of the
safety-critical world, and sector-specific regimes reduce to the same three
questions (what does the tool claim, what happens if it is wrong, what did you
stop doing because of it). Where a sector adds its own tool-validation
expectation, the package in §11 answers it in whatever form it arrives.

---

## Executive summary

stelling is a verifier. That single fact rearranges everything a documentation
architecture has to do, because **a verifier is not the same kind of object as
the software it verifies**, and the standards know it.

A library that ships inside a product is assessed as a *component*. Its defects
become the product's defects: a wrong number propagates, and the integrator's
job is to bound the damage. The documentation that serves it is a component
dossier — what it computes, where it has been validated, what is known to be
broken.

A verifier ships nothing into the product. It sits in the development pipeline
and emits one thing — **verdicts** — and its defects do not become wrong
numbers. They become **misplaced confidence**. A green verdict on a false
property does not corrupt an output; it removes a test, ends an investigation,
closes a review. **The harm from a verifier arrives at the moment someone stops
looking, and it arrives disguised as good news.** No amount of downstream
runtime monitoring catches it, because there is nothing anomalous to catch: the
defect is in the engineer's belief, not in the program's state.

Three consequences organise this document.

**1. The verdict is the product, and its scope is the safety artifact.** A
verdict that says less than it could is merely weak. A verdict that says more
than it earned is the defect — even when the underlying mathematics is right,
because the over-claim is what licenses the removal of work. stelling's primary
safety artifact is therefore not the proof. **It is stelling's own honesty about
what it did not prove**, carried in the verdict itself rather than in a document
somebody might not read.

**2. Red is a fact; green is a claim.** This is not an accident of engineering
maturity — it is the shape of the quantifier. A falsified verdict is an
*existential* claim, and its witness is a concrete input: evaluate the assumes on
it to confirm it lies in R, feed it to the jitted function, watch the assertion
fail. Both halves are executable, which makes the witness checkable by someone
who does not trust stelling at all — **stelling could be a random number
generator and a witness that survives both checks would still stand.** A verified
verdict is a *universal* claim, and **no execution of the program can confirm
it**, because you cannot run all of R. It rests instead on a chain of eight links
(§2.3), **five of which — 2, 4, 5, 6 and 7 — have nothing behind them today that
could license a green verdict** (§2.3.1), one of which — hardware — no roadmap
item will ever close, and the rest of which are held by refusals narrower than
the links they sit on. *This sentence read "three of which — 4, 5 and 6 — have
nothing but a sampling budget behind them today, a fourth of which has that plus
one constraint of use". A sampling budget is not something behind a link: the
sampler can only refute, so a budget spent and quiet leaves the link exactly
where it was. The count changed because the quantity changed, not because the
tree did.* Three of the founding
roadmap's "hard and strategic" items close a link each; the rest are capability,
not trust. **That subset of the roadmap is a trust-debt schedule** (§2.3).

The asymmetry survives all the way down. Link 8 says a witness check runs on
hardware that might lie — but **red is re-runnable and green is not**: a witness
is a finite artifact you can re-check on another machine, another backend, next
week, and each replication is an independent draw. There is no second sample of
"∀x ∈ R".

**3. The classification is the user's, and it turns on one question.** Under
every framework surveyed in §12, stelling is a verification tool whose
malfunction can *fail to reveal* a defect but cannot *introduce* one — IEC
61508-3 class T2, DO-178C criterion 2 or 3, ISO 26262 tool impact TI1 or TI2.
Which of each pair applies is not stelling's to declare. It is determined by
what the user *stops doing* on the strength of a verdict. From this falls the
document's single most useful rule, stated to users in exactly these words:

> **If stelling's output does not cause you to do less of anything else, you owe
> no qualification argument. The moment a green verdict removes work, you own
> one — and here are the artifacts that support it.**

This is why the red direction is exempt under the two frameworks that offer an
exemption (§1.3): DO-178C §12.2.1 and the ISO 26262 TI1 argument both turn on
whether the tool's output is verified or relied upon, and a checked witness is a
verified output. **The bug-finding half of stelling asks those two frameworks for
nothing.** IEC 61508-3 offers no such exemption and its §7.4.4.4 evidence
expectation stands regardless — so this is a real result under two frameworks,
not a universal pass, and §1.3 says which is which. It is not a marketing claim;
it is a consequence of what a witness is.

**What is cheap now and impossible later.** *Written when the repository was
pre-Stage-0 — the IR and the jax boundary existed and the first verdict did
not.* **That stopped being true before `v0.1.0` and the sentence stood for
another five weeks.** On this tree `stelling.__version__` is `0.2.0.dev0`,
`v0.1.0` is a tag, the suite is thousands of tests, and verdicts are minted,
stamped and routed through a solver portfolio. The three things below were fixed at the
moment the paragraph describes, which is why the paragraph is kept: it is the
record of a decision taken in time, not a description of the tree. Every one of
them can still only be *retrofitted* at the cost the table gives, so the
argument stands where the tense did not:

| | Why it cannot be retrofitted |
|---|---|
| **The stamp is a type invariant, not a convention** (§2.4, §10.2) | `SOUNDNESS.md` already specifies what every verdict must carry. Making it structural — a `Verdict` that cannot be constructed without its environment — is a one-day decision now. Ship one unstamped verdict and it is **permanently unauditable**: you cannot go back and learn which solver options produced it. Later stamp *growth* is survivable — §7.3's REVIEW finding exists for it. Stamp *absence* is not. |
| **Evidence retention starts before there is evidence** (§6.2, §10.4) | "History of successful use" is the only qualification route a solo project can support unaided (ISO 26262-8 method 1a; IEC 61508-3 §7.4.4.4). A history is a *record*. CI that prints a green check and discards the corpus results has produced no history. **You cannot backfill a usage record for a release that has already shipped.** |
| **The ledger's scope predicate is executable, not prose** (§7.2) | `SOUNDNESS.md` promises to state "which prior verdicts are retroactively invalid, characterized as precisely as we can." *As precisely as we can* is prose hedging, and prose cannot be evaluated against a stored verdict. Write soundness event #1 with a machine-checkable predicate and `stelling audit` is possible forever. Write it as prose and it never will be — nobody goes back. |

The rest of this document is the architecture those three imply, plus the
documentation tiers, semantics standard, evidence architecture, and
qualification package that hang off them.

---

## Table of contents

1. [Position: stelling in the vocabulary of the standards](#1-position-stelling-in-the-vocabulary-of-the-standards)
2. [The verdict is the product](#2-the-verdict-is-the-product)
3. [Documentation structure](#3-documentation-structure)
4. [Boundary and positioning documents](#4-boundary-and-positioning-documents)
5. [Semantics documentation standard](#5-semantics-documentation-standard)
6. [Evidence architecture: verifying the verifier](#6-evidence-architecture-verifying-the-verifier)
7. [The soundness ledger](#7-the-soundness-ledger)
8. [Versioning, stability, and what a pin buys](#8-versioning-stability-and-what-a-pin-buys)
9. [Contributor standards](#9-contributor-standards)
10. [Code-embedded hooks](#10-code-embedded-hooks)
11. [The qualification package](#11-the-qualification-package)
12. [Standards mapping](#12-standards-mapping)
13. [What stelling is for: two questions §12 does not ask](#13-what-stelling-is-for-two-questions-12-does-not-ask)
14. [Configuration management and reproducibility](#14-configuration-management-and-reproducibility)
15. [Process compatibility](#15-process-compatibility)
16. [Ecosystem](#16-ecosystem)

**Appendices**

- [A: Landscape — what comparable tools do about qualification](#appendix-a-landscape--what-comparable-tools-do-about-qualification)
- [B: Implementation priority, mapped to the founding stages](#appendix-b-implementation-priority-mapped-to-the-founding-stages)
- [C: Relationship to existing documentation](#appendix-c-relationship-to-existing-documentation)
- [D: The qualification roadmap](#appendix-d-the-qualification-roadmap)
- [E: Repository setup checklist](#appendix-e-repository-setup-checklist)
- [F: Ecosystem bootstrap checklist](#appendix-f-ecosystem-bootstrap-checklist)

---

## 1. Position: stelling in the vocabulary of the standards

### 1.1 Tool, not component — the inversion

Every functional-safety framework distinguishes the software that *is* the
system from the software that *builds or checks* the system. The second class is
governed by a separate, smaller, and differently-shaped set of requirements,
usually called tool qualification or tool confidence.

| | A component (e.g. a solver library linked into the product) | A tool (stelling) |
|---|---|---|
| **Ships into the product?** | Yes | No |
| **Failure mode** | Wrong value at runtime | Wrong *belief* at design time |
| **When the harm lands** | When the value is used | When someone stops looking |
| **Detectable downstream?** | Sometimes — bounds checks, monitors, plausibility | **No.** Nothing at runtime distinguishes "tested and correct" from "wrongly believed correct" |
| **Governing requirements** | Component assessment, integration verification, anomaly evaluation | Tool classification and, if warranted, tool qualification |
| **The dossier answers** | What does it compute, where is it valid, what is broken | What does it claim, what happens if the claim is wrong, what did you stop doing because of it |

The last row is the whole difference. A component dossier is about *outputs*. A
tool dossier is about *claims and their consequences*.

**The corollary that matters most:** the single worst artifact stelling could
ship is not a crash, and it is not an "unknown." It is a **transfer function
declaring the tier `sound` without an argument for the tier**. A crash is loud.
An unknown is a work item and says so. A false soundness claim is silent,
propagates into every verdict downstream of that primitive, and is discovered —
if ever — by the user's field failure. §5.1 makes the tier claim the most
heavily reviewed thing in the repository for exactly this reason.

### 1.2 The classification is the user's, not ours

stelling cannot declare its own tool class, TCL, or TQL, for the same structural
reason a component cannot declare its own safety class: **the classification is a
property of the use, not of the artifact.** What stelling can do is state its
shape honestly and hand the user the decision rule.

| Framework | Where stelling sits | What decides the rest |
|---|---|---|
| **IEC 61508-3 §7.4.4** | **Class T2** — "supports the test or verification of the design or executable code, where errors in the tool can fail to reveal defects but cannot directly create errors in the executable software." stelling emits no code and touches no build artifact, so a verdict cannot introduce a defect. **The exception is real and named below.** | The SIL of the item, and the weight placed on the verdict, drive the §7.4.4.4 evidence expected. |
| **ISO 26262-8 §11** | **TI1 or TI2**, undetermined. TI1 requires an argument that no malfunction of the tool can introduce *or fail to detect* errors in the item. That argument exists exactly when the verdict changed nothing you would otherwise have done. | Whether a verdict removed work. TI1 → TCL1 → no qualification, whatever TD says. TI2 → the TD (error-detection) leg decides: TD1 → TCL1, TD2 → TCL2, TD3 → TCL3 (§12.2) |
| **DO-178C §12.2 / DO-330** | **Criterion 3** by default ("could fail to detect an error, within the scope of its intended use"); **criterion 2** if the verdict is used to justify eliminating or reducing another verification or development process. Never criterion 1 — stelling produces nothing that becomes airborne software. | The same question, in DO-178C's own words. §12.2.1 additionally exempts tools whose output *is verified* — see §1.3. |
| **EN 50128 §6.7** | **Class T2**, on the same reasoning as IEC 61508-3. | SIL of the item; the ISA's expectations for tool evidence. |

#### The T2 exception: certified parameter ranges

The T2 argument above rests on stelling emitting nothing that reaches the
executable. **One roadmap item breaks that**, and it must be flagged before it
ships rather than after an assessor finds it.

`design/founding.md` lists **certified parameter ranges** (Medium; and a Stage-0
stretch goal): *"Bisect on an assumed bound to output a* number *— stable step
size, CFL constant, relaxation factor — not just a verdict."* A number a user
copies into their code is an output that **indirectly contributes to the
executable**, which is the T1/T2/T3 boundary in IEC 61508-3's own words. A tool
that hands you `dt_max = 0.0417` and a tool that tells you your `dt` is safe are
different tools under the standard, however similar they look in a terminal.

This is not an argument against the feature — it is the feature `founding.md`
correctly identifies as a differentiator. It is an argument for **naming the
change in class when it arrives**:

- A bisection result rendered as *"this bound is certified"* — a verdict about a
  value the user chose — stays T2.
- The same result rendered as *"use this value"* — a value stelling chose — is
  arguably T3, and T3 is where the rigour actually lives at SIL 3/4.
- **The distinction is in the rendering and the user's workflow, not in the
  mathematics**, which is precisely §1.2's thesis biting the tool that stated it.

When certified parameter ranges ship, `docs/qualification/constraints_of_use.md`
must state the distinction, and the positioning statement's flat "class T2" must
acquire the exception. Recording it here costs nothing now and prevents a
positioning statement that quietly becomes false.

**Rationale.** Refusing to state a TCL/TQL is not evasion — it is the only
correct answer, and stating one would be a category error that an assessor would
(rightly) reject. The failure mode this avoids is the one every commercial tool
vendor's marketing page commits: "qualified for ASIL D" is never true of a tool
in isolation; it is true of a tool *plus a use case plus an argument*.

### 1.3 The rule: what you eliminate determines your burden

This is the most useful sentence stelling can say to a safety engineer, and it
should appear verbatim in `README.md`, in `docs/qualification/positioning.md`,
and in the qualification package.

> If stelling's output does not cause you to do less of anything else, stelling
> is not in the safety argument and needs no qualification. The moment a green
> verdict removes work — a test you no longer write, a review you no longer do,
> a hazard you no longer mitigate — the verdict is load-bearing, and you own a
> qualification argument for it.

Two frameworks reach this by their own routes, which is why it is worth stating
as a rule rather than an opinion:

- **DO-178C §12.2.1** requires qualification when a process is "eliminated,
  reduced or automated by the use of a software tool **without its output being
  verified**." Verify the output and the requirement does not arise.
- **ISO 26262-8 §11.4.5.2** selects **TI1** when there is an argument that no
  tool malfunction can introduce or fail to detect errors in the item. If the
  item is exactly what you would have shipped anyway, the argument is available.

**The red direction is exempt under both.** A falsified verdict's output is a
concrete witness, and stelling checks it — evaluate the assumes on it to confirm
it lies in R, then run the *jitted* function and confirm the assertion fails
(§2.2). That check **is** the verification of the tool's output that DO-178C
§12.2.1 asks for, and it is performed by executing the real artifact, not by
stelling's own reasoning. And a found bug that you then fix and re-verify by your
existing process has not removed any work — it has added some. TI1 is arguable on
the same facts.

**Two frameworks, not all of them.** IEC 61508-3 has no verified-output
exemption. A T2 tool's §7.4.4.4 evidence expectation does not lapse because its
output happens to be checkable, and §12.1 rates that evidence base *substantial
at v0.1*. So the honest scope of this result is: **the red direction asks DO-178C
and ISO 26262 for nothing, and asks IEC 61508-3 for the same T2 evidence
everything else does** — which is a real and useful asymmetry, and not a pass.

Within that scope: **stelling used purely as a bug-finder carries no
qualification burden under DO-178C or ISO 26262.** This is not a loophole; it is
what it means for a witness to be independently checkable. It is also the honest
reason `design/founding.md` scopes Stage 1 as "the wedge: index safety on real
code" — the wedge is the half of the tool that needs least permission to be
useful.

**What this rule is not.** It is a *rule for framing the argument*, not a
guarantee that an assessor accepts it. TI/TD determinations belong to the user
and their assessor; some will push back on any TI1 claim for a tool that touched
the verification plan at all. The artifacts in §11 exist to support the argument
in either direction. Documentation must never suggest that stelling has already
made this determination on the user's behalf.

**A note on attribution.** `design/founding.md` requires replay at Stage 0 and
frames it as *"the encoder's unit test"* — a testing practice. This document asks
for something stronger: replay as a **construction invariant** (§10.2), so that
no falsified verdict can exist without having been checked. The distinction is
the whole argument here. A tool that *tests* that its counterexamples replay has
good hygiene; a tool where an unchecked counterexample is unrepresentable has a
verified output in DO-178C's sense. That is an addition to `founding.md`, not a
reading of it.

### 1.4 The layered responsibility model

| Responsibility | Owner | Evidence |
|---|---|---|
| The verdict means what it says (scope, tiers, arithmetic, trust boundary all stamped) | **stelling** | The verdict artifact (§2.6); `SOUNDNESS.md` |
| Transfer functions are as sound as their declared tier | **stelling** | Tier arguments (§5.1). *Differential testing stood in this cell and does not belong in an Evidence column: it can refute a tier and can never establish one (§6.1, §2.3.1). The argument is the evidence; the testing is the search for its refutation* |
| Witnesses are checked — in R, and the assertion fails under `jit` | **stelling** | A construction invariant, not a test (§10.2) |
| Soundness events are logged, scoped, and mechanically queryable | **stelling** | The ledger (§7); `stelling audit` |
| Coverage — which primitives fell to ⊤, where precision died | **stelling** | The coverage artifact (§6.3) |
| The harness expresses the intended property | **User** | Positive controls, cover obligations (§10.8) |
| The harness traces the code that actually ships | **User** | Query hash vs. current trace; stale-verdict detection (§10.6) |
| Tool classification (T2/TI/TD/TCL/criterion/TQL) | **User** | §12 mappings; their assessor |
| The qualification argument, if a verdict is load-bearing | **User** | The qualification package (§11) as input; the argument is theirs |
| The ℝ ⇒ float gap, until it closes | **User** (named in every verdict) | `arithmetic` field; `docs/semantics/arithmetic.md` |
| The traced ⇒ compiled gap, until it closes | **User** (named in every verdict) | `trust_boundary` field; `docs/qualification/trust_boundary.md` |
| The solver is correct on this query | **Shared, unresolved** | **Nothing today** — certificates eventually (§11.3). *This cell read "cross-solver agreement today". Agreement is not evidence (§6.1, §11.2): two backends that agree have failed to disagree, which is a fact about the pair. The portfolio is real, it runs, and it belongs in the row above as a refutation channel — not here, where the column is headed Evidence* |
| Deciding a verdict is good enough for a given SIL/ASIL/DAL | **User and their assessor** | Everything above |

The last four rows are the honest ones and must never be softened. Two of them
(ℝ ⇒ float, traced ⇒ compiled) are gaps stelling does not close today and says
so in the artifact itself — **the way to make an unclosed gap safe is not to
close it but to name it somewhere it cannot be forgotten**, which is why they
are verdict fields and not documentation prose.

### 1.5 What open source and no certified process cost — and what they buy

stelling is a solo open-source project. It does not operate under a certified
development process, and it will not claim to. §15 is the full treatment; the
positioning-level facts are:

- **Full inspectability.** An assessor can read every line, including the
  transcription boundary, the encoder, and every transfer function's tier
  argument. There is no black box to argue about. For a tool whose entire value
  is trustworthiness, this is worth more than for a component.
- **The trust chain is machine-verifiable at the supply-chain layer already.**
  REUSE compliance, DCO sign-offs, PyPI Trusted Publishing with PEP 740
  attestations (`SECURITY.md`) bind each release to the commit and workflow that
  built it. This answers the configuration-management half of every framework's
  tool requirements (§14) without further work.
- **Apache-2.0, nothing vendored.** No copyleft obligation reaches a user's
  toolchain; there is no NOTICE to propagate because nothing is vendored
  (`README.md`).
- **Nothing is vendored, and the solver is never absorbed.** z3 and cvc5 are
  optional extras the user installs, or — for cvc5 — an external binary reached
  over SMT-LIB 2 (`STELLING_CVC5`). Be precise about the two transports, because
  §8.4 depends on the distinction: the **PyPI wheels are in-process bindings**,
  so a wheel-transport verdict runs the solver inside stelling's process; only
  the **external binary** is a separate program, and `README.md` scopes "nothing
  links into stelling" to that case specifically. What holds in *both* cases is
  the fact that matters here: stelling vendors no solver, distributes no solver,
  and takes no licensing position from one — and, more importantly, **the solver
  is a distinct configuration item the user's dossier must identify separately**
  (§11.2), whichever transport carried it. stelling's job is to make that
  identification exact, not to absorb it.
- **What it costs:** of the four qualification methods ISO 26262-8 offers,
  stelling can support *increased confidence from use* (method 1a) directly, and
  can supply inputs to *validation of the software tool* (1c). It cannot support
  *evaluation of the tool development process* (1b) or *development in
  accordance with a safety standard* (1d) — those require a certified process
  stelling does not have. §15 says which artifact serves which method, and does
  not pretend the gap is smaller than it is.

---

## 2. The verdict is the product

Everything else in this repository is packaging. The verdict is what leaves the
tool and enters someone's reasoning, and it is the only thing an assessor will
ultimately look at.

### 2.1 The verdict taxonomy

| Verdict | Means | Kind of claim | Independently checkable? |
|---|---|---|---|
| **falsified** | ∃x ∈ R with B(x) = false, *and here it is* | Existential, witnessed | **Yes** — replay the witness under `jit` |
| **verified** | ∀x ∈ R, B(x) = true, subject to the stamp | Universal | **No** (see §2.2) |
| **vacuous** | R appears empty; no x satisfies the assumes | Claim about R | Partly — a witness in R would refute it |
| **unknown** | stelling did not decide (precision died, timeout, unsupported) | No claim | N/A — it asserts nothing |

Four rules bind the taxonomy permanently:

1. **`vacuous` is never `verified`.** A vacuously true property is the classic
   silent failure of assertion-based verification and it looks exactly like
   success. The Stage-0 vacuity guard — a sampler must exhibit at least one
   point satisfying R — is not a nicety; it is what makes `verified` mean
   anything. (`design/founding.md`, Stage 0.)
2. **`unknown` is a work item, not a failure, and never a pass.** Design
   commitment 4 makes ⊤ the default for unknown primitives so the tool never
   crashes; the coverage report turns that into a work item. But **`unknown`
   must not exit 0** in a CI context by default (§10.9), because the whole
   failure mode of this tool is people reading absence of red as green.
3. **A verdict with a heuristic transfer in its chain is not a proof, and must
   not render as one.** Tier aggregation is by weakest link (§5.1). A single
   heuristic transfer demotes the entire verdict, and the rendering must say so
   rather than printing a bare `VERIFIED`.
4. **A checked witness beats a `verified` verdict, always** — and when stelling
   produces both for one query, the correct output is neither. It is **"the chain
   broke here, and these are the candidates"** (§2.5), because the two claims are
   about different objects and locating the fault takes more than noticing the
   disagreement.

### 2.2 Red is a fact; green is a claim

The asymmetry is not a maturity gap that better engineering closes. It is the
shape of the quantifier:

> **An existential claim can be confirmed by running the program. A universal
> claim cannot — no execution confirms it, because you cannot run all of R.**

A `falsified` verdict's witness is a concrete array, and checking it has **two
halves that must both be executed**:

1. **Evaluate the assumes on the witness.** Is it actually in R? This half is
   skippable-looking and is not skippable. Because ⊤ is the default for unknown
   primitives (commitment 4), **the encoded region R′ over-approximates R** — so
   a solver model may satisfy R′ and lie outside R, and an input outside R that
   fails the assertion is not a counterexample to anything. `design/founding.md`
   Stage 0 specifies only the second half ("call the *jitted* function → confirm
   the assertion actually fails"); this document requires both, and both are one
   line each.
2. **Run the jitted function.** Does the assertion fail?

Pass both and the check required no trust in stelling's encoder, its transfer
functions, its solver, its treatment of floating point, or its model of what XLA
does — because **the oracle is the compiled program itself**, on the far side of
every one of stelling's trust boundaries. Skip the first and stelling can report
a "counterexample" that is an artifact of its own over-approximation, which is
the most humiliating output a verifier can produce.

A `verified` verdict has no executable check at all, because you cannot run all
of R. Its truth rests entirely on the chain in §2.3. This is why:

- the red half of the tool is exempt from two of the three frameworks (§1.3) and
  can be useful on day one;
- the green half requires either trusting the chain or **checking a proof**, and
  a machine-checkable proof certificate is therefore not an auditor's luxury
  item on the far end of the roadmap — **it is the only mechanism that can ever
  make a green verdict independently checkable at all** (§11.3). Note what that
  concedes: a certificate is a finite artifact for a universal claim, so the
  quantifier does not forbid checkability. It forbids checkability *by
  execution*. A proof is a different kind of object, which is exactly why it
  costs so much to get one;
- **the only oracle over the program's actual behaviour is the jitted
  function.** Every technique in §6.1 that adjudicates a claim about *what the
  code does* is an application of it: witness checking, differential testing
  against concrete corpus runs, and fuzzing a *verified* query (§6.1) are three
  uses of one oracle. The exceptions prove the rule and are worth naming: the
  round-trip test's oracle is the IR (it checks serialization, not behaviour);
  cross-solver agreement has no oracle at all (the solvers only check each
  other, so agreement is not evidence); and a proof checker's oracle is the
  proof, not the program. **None of those three can tell you what the program
  does** — which is why the corpus is load-bearing rather than decorative.

### 2.3 The trust product — the eight links of a green verdict

For a `verified` verdict to mean *"this property holds of the program that
runs"*, eight links must all hold. **The "Planned defence" column is planned**,
and it was written when the repository was pre-Stage-0, so every cell in it was
a commitment rather than a capability. **That column has not been re-derived
against the tree since**, and this document does not get to claim it has: some
of those defences now exist, some do not, and which is which is not settled
here. The qualification package's copy of this table (§11.1) must carry the
*actual* state at each release rather than this one — and until it does, read
this column as the plan of record and not as an inventory.

**AND "DEFENCE" IS THE WRONG WORD FOR MOST OF WHAT IS IN IT.** Held to §2.5's
capability table — ***it is a search; absence of evidence is not evidence*** —
the cells sort into four kinds, and only one of them defends anything:

- a **licence** — a mechanism whose success is a reason to believe the link
  holds. Two shapes are available and no others: a **refusal** that cannot be
  talked past (transcription that raises on a param it has no rule for has
  *proved* that no such param was silently dropped), and a **checkable
  artifact** (a proof certificate an independent checker validates). A licence
  is always narrower than the link; the question to ask of one is *what exactly
  did it prove.*
- a **refutation channel** — a mechanism that can only fail. Its firing is
  decisive; **its null result licenses nothing**, and recording that null result
  as though it did is the defect §2.6 now refuses. Fuzz-on-verified, cross-solver
  agreement, the oracle property, every mutation battery and every containment
  sweep are of this kind, and so is every technique in §6.1 whose oracle is
  execution.
- a **constraint of use** — not a defence of the link but a narrowing of the
  claim, so the link's failure mode falls outside what was asserted. Pinning the
  precision (§4.2) is the example, and it earns more than the label suggests:
  it converts link 7's *silent* failure into a compile-time error, which closes
  the silent half and nothing else.
- a **naming** — a stamp field that discloses the gap and closes none of it
  (`arithmetic`, `trust_boundary`, `precision`, `vacuity`).

A column of refutation channels read as a column of defences is how a trust
argument comes to report four links as covered when every one of them is bare.
**The kind is marked in every cell below**, and §2.3.1 re-derives them against
the tree, which is the inventory this table is not.

| # | Link | The claim | If it fails | Planned mechanism, and its kind | What would close it | Phase |
|---|---|---|---|---|---|---|
| 1 | **Harness** | R and B say what the engineer meant | Verified answers a question nobody asked | Vacuity guard — **R ≠ ∅ only** (*licence*, for exactly that much: a point in R is exhibited, not searched for and missed) | Positive controls; cover obligations; review (§10.8) | 1 |
| 2 | **Transcription** | the IR is the jaxpr | Everything downstream concerns a different program | Raise-on-unknown-param (*licence*, narrow: no param was dropped for want of a rule); per-primitive pinned traces (*refutes*); **fuzz-on-verified** (§2.5) — *refutes* | Property-based transcription tests over the corpus | 1–2 |
| 3 | **Query construction** | (B, R) is the property | Verified concerns the wrong formula | Witness checking (*licence* — and only for a `falsified`; it says nothing about a green one); **fuzz-on-verified** — *refutes* | Positive controls. *Cross-engine agreement stood here and closes nothing — it is a refutation channel and belongs in the column to the left* | 1 |
| 4 | **Transfer soundness** | each transfer is as sound as its declared tier | A `sound` tier that isn't ⇒ **false verified** | **fuzz-on-verified** only — *refutes*, so **no defence: a chance of being caught** | Differential testing; tier arguments; per-transfer proofs | 3 |
| 5 | **Solver** | the solver is right on this query under these options | False UNSAT ⇒ **false verified** | **fuzz-on-verified** only — *refutes*, so **no defence** | **Proof certificates + independent checker** — the only thing that closes it. *Cross-solver agreement stood first in this cell; it is real, it runs, and it refutes. It does not close a link* | 3 / 5 |
| 6 | **ℝ ⇒ float** | the margin absorbs rounding | Proven over ℝ, false in float | **fuzz-on-verified** only (*refutes*); named in the verdict (*naming*), carried by the user — **no defence** | FPTaylor-style per-step bounds; the margin-absorption argument | 5 |
| 7 | **jaxpr ⇒ executed program** | the jaxpr denotes the computation XLA runs on this device | Verified the traced program, ran a different one | **Pinning the precision** (§4.2) — a *constraint of use*: it narrows the claim rather than defending the link, and closes the **silent** half by making the unhonourable case a compile error; **fuzz-on-verified**, on the device that will run it — *refutes* | Translation validation to HLO | 2 (pin) / 5 |
| 8 | **hardware** | the machine computed what it was told to compute | Verified and compiled correctly, executed wrongly | *Nothing, and nothing will* — see below | **Unclosable.** Redundancy and re-execution are mitigations, not closures | — |

And the red direction, in full:

| # | Link | Checked by |
|---|---|---|
| 1 | The witness lies in R, and makes the assertion fail under `jit` | **Both halves, every time, as a construction invariant** (§2.2, §10.2) |

**This table is the qualification package's core argument** (§11.1) and the most
useful single artifact in this document. Six readings of it:

- **Link 7 is not the bet Kani makes, and the difference is the whole of §4.3.**
  Earlier drafts of this document said *"Kani makes the same bet on rustc's
  backend."* That was wrong and it flattered us. LLVM does not reassociate floats
  without a fast-math flag, and rustc does not set one: MIR's float semantics
  survive to machine code as **a contract the backend documents and honours**, so
  Kani is entitled to lean on it. **XLA offers no equivalent, and the reason is
  not that XLA is worse — it is that the jaxpr does not denote a unique numerical
  program.** A `dot_general` at f32 is not a computation until you know the device
  and what XLA chose on it. Verified against jax 0.10.2 and 0.11.0 on 2026-07-16:
  `jax.config.jax_default_matmul_precision` is `None` — unset, platform's choice —
  and `lax.Precision`'s own docstring calls itself *"the **device-dependent**
  `precision` argument"*, then spells out what that costs: `DEFAULT` on TPU
  *"performs float32 computations in bfloat16"*; on GPU it *"uses tensorfloat32 if
  available (e.g. on A100 and H100 GPUs), otherwise standard float32 (e.g. on V100
  GPUs)."* **Same jaxpr, same flags, two GPUs, two numerics.** The names make the
  point by themselves: `lax.Precision('bfloat16_3x')` and
  `lax.Precision('tensorfloat32')` **both evaluate to the same member, `HIGH`** —
  you can ask for bfloat16_3x and get tensorfloat32, because they were never two
  requests. The link-7 analogue of `SOUNDNESS.md`'s line about solver options:
  **one jaxpr, three devices, three numerics.**
- **Link 7 has a partial defence, and it is a constraint of use, not a roadmap
  item.** Pin the precision (§4.2). This is worth more than it sounds: JAX ships
  *two* mechanisms and only one of them is a contract. `precision=` is a **hint** —
  `HIGH` on a GPU means tf32 *"where available, otherwise float32"*, silently. But
  `lax.DotAlgorithmPreset` is a **contract**, and the route from a preset member
  to that guarantee is `DotAlgorithm`'s own docstring, which says *"Support for
  these algorithms is platform dependent, and using an unsupported algorithm
  will **raise a Python exception when the computation is compiled**"* and then,
  in its very next sentence, says which algorithms it means: they *"are listed
  in the `DotAlgorithmPreset` enum"*. `F32_F32_F32` is a real member of that
  enum. *(TWO repairs here, and the second is the one worth reading. The
  quotation was first attributed to the preset enum's own docstring, and it is
  not in it on either pinned series: 0.10.2 and 0.11.0 both carry it on the base
  class and give the preset enum a docstring that points at the base class. The
  repair for that then asserted that a preset member **IS** a
  `lax.DotAlgorithm`, and that is false on both series. Driven:
  `isinstance(lax.DotAlgorithmPreset.F32_F32_F32, lax.DotAlgorithm)` is `False`,
  `issubclass(lax.DotAlgorithmPreset, lax.DotAlgorithm)` is `False`, the preset
  is an `enum.Enum` where `DotAlgorithm` is a `NamedTuple`, and of
  `DotAlgorithm`'s seven fields the member exposes exactly one,
  `accumulation_type`. jax's own text invites the error —
  `DotAlgorithmPreset.__doc__` calls itself *"a named set of `DotAlgorithm`
  objects"*, which is fair as description and wrong as typing — but a false
  MECHANISM replacing a false ATTRIBUTION, in the paragraph whose whole subject
  is getting an attribution right, is worse than what it replaced. The
  conclusion survived both times; it now travels by the route the base class
  states itself.)* So the constraint of use is not "prefer HIGHEST" — it is
  **"state the algorithm, not the preference, and let it fail loudly on a device
  that cannot honour it."** Link 7's Planned-defence cell is not *Nothing*.
- **Four links are bare, and the fuzzer is why that was not obvious.** Read off
  the table above: links **4, 5 and 6** are the rows whose mechanism cell is
  *"**fuzz-on-verified** only"*; link **7**'s cell leads with the precision pin,
  which narrows the claim rather than defending the link; and link 8's is
  *"Nothing, and nothing will"*. So **four** links are unchecked by any
  *dedicated* mechanism before Phase 3 — **and all four of them are bare.**

  ***That last clause read "but none of the four is bare, because
  fuzz-on-verified (§2.5, §6.1) points at all of them at once", and it is the
  error this whole document exists to catch other people making.*** §2.5's
  capability table, one screen down, bars the sampler from `verified` on the
  ground that **absence of evidence is not evidence**. A mechanism that can only
  refute does not stop a link being bare — it gives a broken link a chance of
  being noticed. Pointing one such mechanism at four links does not cover four
  links; it gives four links one chance each, and a chance is not a defence.
  Nor is it a small correction: this sentence is what the qualification package
  would have carried into §12.2's TD row, where "the link has a defence" and
  "the link has a detector that has never been shown to detect anything on it"
  are the difference between two confidence levels. **Four bare links in a
  trust argument are a disclosure, and §2.3.1 makes it.**

  **Two counts, two quantities, and this is the one place they must not be
  conflated.** *Four* is the number of links with no dedicated mechanism
  before Phase 3 — 4, 5, 6 and 7, read straight off the table. *Five* is the
  number with **no licence for a green verdict** — 2, 4, 5, 6 and 7 — which
  §2.3.1 derives against the tree, and which includes link 2 because the
  mechanisms in its cell either refuse (a licence narrower than the link) or
  refute (no licence at all). The paragraph below is about a bullet that used
  three numerals for *one* quantity; these are two quantities and each is
  named where it is used.

  *This bullet said* **five** *and then enumerated* **four**, *in one sentence,
  and the executive summary repeated the five where a reviewer meets it first.
  No reading of the table yields five: the "fuzz-only" rows are three and the
  "no dedicated mechanism" rows are four. Three numerals for one quantity, one
  of them contradicted by the bullet directly above it.*

  Link 4 is the one **stelling owns entirely and can close by
  itself** — differential testing needs no research, only work, and even then it
  buys a refutation channel rather than a licence, which is the ceiling on every
  route to link 4 short of a per-transfer proof — which is why it outranks the
  others despite being the least glamorous. Links 6 and 7 the *user*
  carries (§1.4); link 5 is shared and unresolved; link 8 is nobody's and stays
  open forever.
- **Part of the roadmap is a trust-debt schedule — not all of it.** Exactly three
  of `design/founding.md`'s ten "Long — hard and strategic" bullets close a link:
  IEEE-754 semantics closes 6, translation validation closes 7, proof certificates
  shrink 5. The other seven — shape generalisation, Pallas, k-induction/CEGAR,
  polyhedral/SOS, coupled-system stability, supermartingales, sharded programs —
  buy **capability**, not trust, and mixing the two categories would misrepresent
  a roadmap this document has no standing to reorganise. The useful claim is the
  narrow one: **those three items are not research ambitions, they are the
  scheduled repayment of debt every green verdict is currently carrying.**
- **The links are not independent, but the coupling is narrower than it looks.**
  Commitment 2 couples 6 and 7 *for reassociation only*: robust invariants with
  slack are insensitive to reordered sums — *"fusion shuffles ulps, robust
  invariants don't care"* (`founding.md`) — so the margin that serves link 6 does
  discharge that much of link 7. **It does not survive a precision selection.**
  bfloat16 against float32 is fifteen mantissa bits, not an ulp; a margin sized to
  absorb f32 rounding absorbs nothing of a silent demotion to bf16, and a
  miscompilation is not a rounding difference either (§4.3). The coupling covers
  reassociation and stops there. Read the table as eight things that must hold,
  not as eight probabilities to multiply.
- **Links 6, 7, and 8 are named, not closed.** 6 and 7 are verdict fields
  (`arithmetic`, `trust_boundary`, `precision`), not footnotes, so a verdict
  cannot travel without them. Commitment 5 already says this — *"we verify what
  traced, XLA runs what compiled, and we say so plainly"*. The architecture's job
  is to make "plainly" mechanical. Link 8 is named here and in the package (§11.1)
  and nowhere else, because there is nothing to stamp: it is a property of the
  machine, not of the run.
- **Link 1 is the user's, and it is the least defended.** See §10.8: the
  vacuity guard catches R = ∅ and nothing else. It does not catch B ≡ true.

#### 2.3.1 What actually defends each link — the column re-derived against the tree

**Derived 2026-08-24 against `main` at `115d771`**, by reading the shipped
modules and the shipped suite. The table above is July 2026's plan and says so;
this is the inventory it told the reader to go and take, and it exists because
the plan column was being read as one. It is dated the way every other empirical
claim in this repository is dated: **re-derive it at each release, and do not
quote it as current without doing so.**

**The result, stated first because it is the finding and not a summary:**
*nothing in this tree licenses a `verified` at links 2, 4, 5, 6 or 7.* Every
mechanism that bears on the truth of those links is a refutation channel, and a
refutation channel that has not fired has said nothing. What differs between the
five links is not whether they are defended — none of them is — but **how many
refutation channels point at them, whether those channels run without being
asked for, and whether any of them has ever been shown capable of firing on that
link.** Those three questions have different answers per link and they are the
honest content of this section.

**And the claim that a single probe covers four links at once is not merely
unlicensed — it is measured false in this tree.**
`tests/test_falsify_wrap_reach.py::test_NEITHER_instrument_reaches_the_jnp_full_door`
records an open, disclosed false `VERIFIED` — an out-of-dtype-range integer
literal narrowed before tracing — that **neither** the falsification probe nor
the overflow tripwire reaches. A defect the tree knows about, has written down,
and cannot detect is the strongest possible answer to *"but the fuzzer points at
all of them at once"*: the fuzzer is pointed at it and does not see it.

**Link 2 — transcription.**

- *Licence, and it is real but narrow.* The transcriber's param whitelist
  (`_jax_compat`'s `_Transcriber.param`) **refuses** rather than drops: a param
  type with no transcription rule raises `ir.UnsupportedParamError`, and so do a
  non-empty mesh, a non-trivial sharding, an unmaterialisable const and a
  non-static dimension. `ir.py` states the asymmetry that makes this the right
  shape — unknown *primitives* are fine, because the registry sends them to ⊤;
  unknown *params* are not, because dropping one changes the semantics of the
  equation it configures. **What it licenses is one sentence:** *no equation in
  this IR lost a param for want of a rule.* It licenses nothing about whether a
  param that was transcribed was transcribed rightly.
  `ir.CANONICALIZATIONS` is the same shape one level up — it licenses *the
  canonicalizations are exactly these*, with a witness apiece in
  `tests/test_ir_canonicalization.py`, so an entry that stops describing the code
  reds and a witness with no entry reds.
- *A licence that does not run where the reader assumes.* `_REQUIRED_PARAMS`
  refuses an equation missing a param jax always supplies — **on the load path
  only, deliberately**, because hand-built IR legitimately omits params. It
  defends a *deserialized* document, not a traced one.
- *Refutation channels that run.* The oracle property in `tests/property/`
  evaluates the obligation **as the user wrote it**, in unbounded Python
  integers, and refutes a `VERIFIED` that has a violating admitted point. It is
  the only instrument in the tree whose ground truth is the *source text* rather
  than an execution, and that is exactly why it reaches link 2: the defect it was
  built around wraps a literal mod `2**bits` *before* tracing, which an execution
  oracle wraps too and then agrees with. Its power is demonstrated rather than
  asserted — `tests/property/positive_controls.py` pins, per property, a commit
  at which it is known to fail, and a property whose control cannot be
  demonstrated does not ship. Its limits are stated by its own README and are
  severe: **integers only**, one declaration, boxes small enough to enumerate
  exactly, and at the per-push budget it is *"a rot detector, not a defect
  finder"* in as many words. The metamorphic properties beside it need no ground
  truth and are stronger than an execution oracle precisely where a defect is in
  the translation. The cross-series differential refutes drift between the two
  tested jax series and **skips** unless a second interpreter is supplied.
- *Refutation channel that is opt-in.* The overflow tripwire, armed as a pytest
  plugin. Armed, it refuses a verdict on a trace it saw narrow a constant — and,
  in its **third state**, refuses one on a trace it could not fully watch, in its
  own sentence: *"none was seen, and none was seen not to be."* That third state
  is this document's doctrine already implemented in shipped code, by an author
  who declined to report an unmade measurement as a clean one.
- *Refutation channel that does not run.* Fuzz-on-verified. As built it is
  `check(..., falsify="sample")`, **default-off** — with the default the module
  is never imported — and no job under `.github/` and no script under `tools/`
  turns it on. Its reach is bounded again by its own fire condition: only an
  **exact** reading may admit a violation, so a program its exact-rational replay
  cannot read through is a program it cannot fire on however false the
  obligation is.
- *Absent.* **Per-primitive pinned traces do not exist.** Nothing in the suite
  compares a live `make_jaxpr` against a committed jaxpr or IR document; what
  exists pins which transfer *name* fired, which is a name binding, not a shape.
  Property-based transcription tests over the corpus — the table's "What would
  close it" — do not exist either. And the IR round-trip is not a transcription
  check: its oracle is the IR, so a transcriber that dropped an equation would
  round-trip perfectly (§6.1 says this already).

**Link 4 — transfer soundness.**

- *No licence, and the reason is one line.* The tier is a **label**. `propagate`
  declares `TIER_EXACT`, `TIER_SOUND` and `TIER_SOUND_LIBM` beside each row of
  its registry and the stamp carries the pairs that fired. **There is no `Tier`
  type and no `TransferMeta` in the source tree**, no `argument_ref`, and no
  registration-time refusal of a tier whose argument does not resolve; those are
  §10.1's Phase-2 items and Appendix E's boxes for them are unchecked. So a tier
  today is the author's assertion, carried faithfully to the reader, with nothing
  behind it the reader can open. The arguments themselves are real where they
  exist — outward-rounded brackets in the interval kernel, exact-rational
  coefficients with outward snaps in the affine refinement, a measured
  per-decline-class error table in `dot_general`'s docstring — but **nothing in
  the tree binds a label to its argument**, which is the entire content of the
  Phase-2 item. `transfer_provenance` is a related trap: it is a stamp field
  whose every entry is the constant `"core"`, so it reads as a provenance and
  records nothing.
- *A shipped mitigation that is a policy, not a check.*
  `verdict.VERIFIED_BARRED_PRIMITIVES` downgrades a `VERIFIED` whose emitted
  slice contains `scatter`, on the stated ground that a *missed* violation would
  mint a false `VERIFIED` with nothing downstream to catch it. It is
  hand-maintained and currently holds one primitive; it is a real, shipped,
  fail-closed narrowing of what green can be claimed on, and it defends nothing
  about the rows it does not name.
- *Refutation channels that run, and this is where the tree is strongest.* The
  mutation batteries — the `pow` and `square` row gauges, the scatter gauges, and
  the `fidelity.gauge` bar they are held to — **do not report an absence. They
  report detection power.** A mutant that survives is a statement about the
  suite, an unexplained survivor is a red, and a gauge's zero is only meaningful
  against the surface it drove, which is why every report carries its scope. That
  is the one shape in this repository that supports a claim stronger than *"we
  looked"*, and it is the shape any TD argument would have to be built from
  (§12.2). Beside it sit the containment gauges, which sample the true value set
  **eagerly and under `jit`** and assert the computed box contains it, and the
  exact-rational containment sweep behind the ledger's S10 entry, whose counts
  are module constants so drift reddens the suite rather than leaving prose to be
  believed. That sweep's own disclaimer is the model for this whole section: it
  is *"a containment BATTERY, not a proof"*, it drives the kernels rather than
  the traced pipeline, and *"it says nothing about transfers it does not name."*
- *Absent, and the gap is a ratio.* Containment gauges exist for a handful of
  primitives against a registry of some eighty-odd rows. There is no
  registry-wide containment sweep, and there is no `tests/differential/`.
  Differential testing against concrete corpus runs — the table's "What would
  close it" — is not built. Note the ceiling even when it lands: **differential
  testing is a refutation channel too. The only route from link 4 to a licence is
  a per-transfer proof**, and that is on no phase in this document.

**Link 5 — the solver.**

- *Licences, and they are about the record rather than the answer.* Several
  mechanisms in `solvers` refuse to emit a verdict over a record that does not
  describe the work: `ProvenanceError` (a spawn counter and an invocation ledger
  maintained from disjoint code sites, deliberately anti-correlated, checked for
  agreement before any escalated verdict emits), the **query-pairing gate** (an
  escalation carrying work must reproduce the `content_hash` of the query it is
  stamped against, and an absent hash is refused too), and the
  **semantics-pairing gate** (an escalation may not be stamped against a
  propagation of the other semantics, in either direction). Together these
  license *this stamp is about this query, under these semantics, and the
  invocations it names are the ones that happened.* That is worth having, it is
  the mechanised form of `ir.py`'s own *"did both solvers see the same query"*,
  and it is not a claim about whether the solver was right.
- *A licence about the region.* After an `unsat`, the same backend is asked
  whether the boxes and axioms alone are unsatisfiable; if they are, the
  discharge is refused as an unsatisfiable assumption rather than reported as a
  proof. Nothing has to be believed across solvers for that reading — it is one
  backend's two answers — and it catches the *relational* vacuity the interval
  detector structurally cannot see.
- *Refutation channel, and it is the best-instrumented one in the tree.* The
  portfolio is **two backends, independently, on the same emitted text** —
  `PORTFOLIO_SIZE` is the constant 2 rather than `len(backends)`, precisely so
  that *"did this verdict get the cross-check the design promises"* is
  answerable — and a sat/unsat disagreement **raises** and is never a tiebreak.
  When fewer than two answered, the verdict carries a portfolio-degradation
  disclosure that says what was lost, in the right direction: *"a discharge is a
  universal claim over the whole declared box, so nothing downstream re-derives
  it the way exact-rational replay re-derives a witness."* Two qualifications
  that matter: escalation is **opt-in** (the solver timeout has no default;
  omit it and the verdict is interval-only), and agreement licenses nothing —
  **agreement is not evidence** (§6.1, §11.2), because two solvers wrong the
  same way on a shared emission is the case it cannot see.
- *Absent.* Proof certificates and an independent checker: nothing in the tree
  requests, stores, replays or checks a proof object. An `unsat` is believed on
  the backends' word, qualified only by the fact that two of them said it. This
  is the only item on this document's roadmap that would turn link 5 into a
  licence (§11.3).

**Link 6 — ℝ ⇒ float.**

- *A licence the table does not have, and it should.* `semantics="ieee"` exists,
  is shipped, and is **format-parametric** — binary64, float32, float16 and
  bfloat16 are each judged in their own format. In that mode obligations are
  judged with rounding modelled rather than in ℝ, so **for the censused
  behaviours the ℝ⇒float gap does not arise, because the verdict is not about
  ℝ.** It is fail-closed at two seams that matter: **IEEE-mode propagation
  refuses solver escalation** (the SMT backends emit over the reals, so
  escalating would prove the ℝ obligation under a float-stamped claim), enforced
  twice from anti-correlated sites; and comparisons or converts it cannot judge
  in the narrower formats decline rather than reuse the binary64 rule.
- *Disclosure that is doing real work.* Every IEEE verdict stamps its
  assumptions: endpoint arithmetic, subnormal indeterminacy (the band is hulled
  with zero so the verdict is sound under **both** flush and gradual), equation
  order, contraction, NaN hygiene — and each of those texts is itself
  format-parametric, after an audit found binary64 sentences being stamped
  verbatim on float16 verdicts where they were false. The `libm_budget` is the
  sharpest of these: it is **declared, not verified**, and the door says so —
  *"a budget smaller than your backend's real error mints a VERIFIED stelling
  cannot catch."*
- *Naming, in the default mode.* Under the default `semantics="real"`, link 6 is
  open exactly as the table says, and the stamp names it — the semantics line
  says in as many words that the traced program's IEEE behaviour is not modelled
  and a predicate can hold in ℝ and fail in floats. Naming is disclosure and
  closes nothing.
- *Absent, and this one is worth correcting elsewhere in this document.*
  **There is no `real-with-margin` mode and no margin.** No margin is computed,
  no margin rides on a verdict, and the string `real-with-margin` occurs in this
  document and nowhere in the source tree; the stamp field is `arithmetic_mode`
  and it names how brackets are computed. §2.4's `arithmetic` row and §2.6's
  artifact are describing a dial position that does not exist yet. FPTaylor-style
  per-step bounds and the margin-absorption argument are absent as the table
  says.
- *So the honest statement is not "link 6 is bare".* It is: **link 6 is bare in
  the default mode, and in the opt-in mode it is replaced by a narrower claim
  that does not need it** — at the price of the solver, which that mode refuses.

**Link 7 — jaxpr ⇒ executed program.**

- *Two licences, both shipped, both narrow, and both only in IEEE mode.* The
  **FMA contraction hull** is the tree's one mechanism that models an XLA
  freedom rather than assuming it away: the contracted value of `a*b + c` is
  computed exactly and hulled with the uncontracted one, so the verdict is sound
  whichever the compiler chose, and a form that cannot be bracketed declines.
  And **intra-equation order freedom declines** — a `reduce_sum` over more than
  two elements, an `integer_pow` past the first power — rather than assuming an
  accumulation order. These are real closures of small pieces of link 7.
- *An assumption, stamped rather than checked.* IEEE mode judges each equation as
  the binary64 operation it names, which assumes the compiler does not
  reassociate *across* equations. The stamp says so, and says it was verified on
  one measured target and is *"a compiler assumption, not a language guarantee."*
- *A shipped partial licence in both modes.* `dot_general` declines to ⊤, with
  the reason quoted into the notes, on an unrecognised precision, a non-`None`
  `out_sharding`, an integral operand with no accumulation type, or an
  accumulation type narrower than an operand. It **admits** `DEFAULT`, `HIGH` and
  `HIGHEST` as equivalent, on the ground that they measure identical on the
  backend it was measured on and that ℝ semantics does not model float rounding
  anyway — with the residue deferred to `device_class`, which is the next bullet.
- *Naming that does not name.* `precision_config` is a live reading of the
  actual `jax_enable_x64` state at trace time and is genuinely informative.
  `device_class` is a required stamp field whose value at every shipped call site
  is the same literal — *"no concrete execution in this verdict"*. That is true of
  the pipeline and it is not what §2.4's precision row promises, which is the
  devices the verdict was asserted for. **And `trust_boundary` does not exist in
  the source tree at all**; it is a field this document specifies and a commitment
  `design/founding.md` states, and it is not yet a thing a verdict carries.
- *Constraint of use, and it is prose.* Pinning the precision (§4.2) is
  documented, argued, and unimplemented: **`DotAlgorithmPreset` appears nowhere
  in `src/` or `tests/`.** Nothing requires it, nothing refuses an unpinned run.
- *Refutation channel that does not run.* Fuzz-on-verified is the only mechanism
  in this architecture that executes the compiled artifact against a green claim.
  It is default-off; it declines on any program its exact-rational replay cannot
  read through; and it carries its own granularity guard, which declines when the
  violation's truth value moves between the equation-at-a-time walk and the whole
  program compiled as one region — a guard that *can only ever decline*.
- *Absent.* Translation validation to HLO, and any programmatic lowering or
  comparison of HLO at all. Also absent: the one-line differential this document
  proposes twice — check a witness on CPU as well as on the accelerator — which
  Appendix E still lists as an unchecked box, and which in any case bears on the
  red direction.

**Where that leaves the trust argument.**

1. **Five links are bare and one of them was double-counted as covered.** Links
   2, 4, 5, 6 and 7 have no licence for a green verdict. Link 6 has an opt-in
   mode that replaces the claim rather than defending it; link 7 has two shipped
   licences that live only inside that mode, which is the mode that refuses the
   solver — **so the strongest link-5 mechanism and the strongest link-7
   mechanisms in this tree are mutually exclusive by construction.** That is a
   fact about the design worth knowing before it is discovered by someone
   assembling a qualification argument.
2. **The refutation channels that exist are worth much more than the plan column
   gave them credit for, and saying so is not a retreat from point 1.** A
   two-backend portfolio that raises on disagreement and discloses its own
   degradation; exact-rational containment sweeps with pinned counts; a
   source-text oracle with a demonstrated control per property; mutation
   batteries that measure their own detection power; a trace gate with a third
   state for *"nobody looked"*; a widen re-check that reports that neither of its
   own outcomes is evidence — none of these is in the July table, and together
   they are why a defect in this tree has a decent chance of being found. **None
   of them is a reason to believe any particular green verdict.**
3. **The qualification package (§11.1) must carry this section's shape, not the
   plan column's**, and a bare link is a **disclosure**, not a work item. It
   belongs beside the policy that governs verdict flips, where a reader deciding
   whether to rely on a green verdict will meet it — not only here, where a
   reader has already decided to read an architecture document. Routing it into
   `SOUNDNESS.md` is that file's owner's call; this section is the input.

#### Link 8: hardware, and why the chain must not stop at 7

Silent data corruption is real and documented at fleet scale: a small but nonzero
population of cores computes the wrong answer, intermittently, under load, without
signalling anything. No roadmap item closes this. Translation validation would
give you a proof that the emitted machine code implements the jaxpr; it would say
nothing about whether the machine ran that code correctly today.

**It is named precisely because it is unclosable.** A chain that terminates at 7
implies that closing 7 reaches *"the property holds of the program that runs."* It
does not. It reaches **"the property holds of the machine code that was emitted."**
Everything after that is physics, and this document's whole method is to name the
gap rather than let the reader assume it away — link 8 is the honest terminus, and
a trust argument that ends at 7 is over-claiming by exactly one link.

**It is also the only link that touches the red direction.** A witness check
executes on hardware; if the hardware lies, the check lies. That looks like a
threat to §2.2's proudest claim and is not, because of the property §2.2 never
had to lean on until now:

> **Red is re-runnable and green is not.** A witness is a finite artifact. Run it
> again, on another machine, on another backend, next week, in front of the
> assessor — every replication is an independent draw against link 8, and they
> converge fast. A green verdict cannot be re-drawn: there is no second sample of
> "∀x ∈ R". You can re-run the *solver*, but that re-runs links 3–5 and tells you
> nothing new about 8, because the same solver on the same query is not an
> independent draw of anything.

So link 8 does not weaken the asymmetry — it **deepens** it. Red survives hardware
faults by repetition; green has no repetition available to it. That is the same
∃/∀ shape as §2.2, arriving one level down, which is a reasonable sign the
distinction is load-bearing rather than rhetorical.

One practical consequence, and it is one line of code: **check witnesses on CPU as
well as on the accelerator.** Different backend, different compiler path,
different silicon — a free differential against links 7 and 8 for the cost of one
extra call. `lax.Precision`'s docstring notes it *"has no impact on CPU backends"*,
which makes CPU the one place a f32 computation means f32 — so a witness that
checks on CPU and on the accelerator has been checked against both a different
machine and a different set of numerics. Appendix B, Phase 1.

#### Why the wedge is the trustworthy half — a sequencing argument

`design/founding.md` picks index safety as Stage 1's wedge on the grounds that it
kills a real bug class with pure integer reasoning. There is a second reason, and
it is independent of the first:

> **Link 7 exposure is ≈ 0 for index safety and high for the float invariants.**

Integer index arithmetic is not reassociated, is not precision-selected, and
gather/scatter clamping is HLO-defined and preserved — the whole apparatus that
makes link 7 hard is an artifact of floating point, and the wedge has none of it.
Nor does it touch link 6: there is no margin to absorb, because there is no
rounding. **The wedge's green verdicts depend on links 1–5 and 8; the float
invariants depend on all eight.**

Which means stelling's first shipped verdicts will be its most trustworthy ones,
and its hardest verdicts will arrive years later, when the machinery to justify
them exists. That is the right order to earn credibility in, and it is worth
recording because it is a reason for the wedge that survives even if the "integer
SMT is easy" reason evaporates.

### 2.4 The stamp is part of the claim

`SOUNDNESS.md` already specifies the stamp's contents. This document adds one
architectural requirement:

> **A verdict without its stamp is not an under-documented verdict. It is not a
> verdict.** Make it a type invariant: a `Verdict` that can be constructed
> without its environment is a bug in the API, not a gap in the docs.

This is the sharpest divergence from the component-dossier pattern, where
metadata is an optional annotation attached to a computation. Here the
assumptions *are* the claim. Dropping them does not reduce information — it
inverts the meaning, because a bare "VERIFIED" asserts strictly more than the
stamped one does.

Fields, per `SOUNDNESS.md` plus what §2.3 requires:

| Field | Why |
|---|---|
| `stelling_version` | The ledger's scope predicates are versioned (§7.2) |
| `jax_version` | Transcription is jax-series-dependent (`TESTED_JAX_SERIES`) |
| `solver`, `solver_version`, `transport` | Wheel vs. external binary; for external cvc5, its `--show-config` feature set |
| `solver_options` | **The full emitted set, never defaults.** `SOUNDNESS.md`: three configs, three engines, one version string |
| `query_hash` | `ir.ClosedJaxpr.content_hash()` — the spine (§2.6) |
| `tiers` | The assumption tier of every transfer function in the chain (commitment 5) |
| `soundness` | Derived: `proof` if the chain is all exact/sound, `heuristic` if any transfer is heuristic (§5.1) |
| `arithmetic` | `real-with-margin` \| `float-exact` — link 6, named |
| `trust_boundary` | `jaxpr` \| `hlo` — link 7, named |
| `precision` | **The precision configuration the verdict assumes**, and the devices it was asserted for. Link 7, named. `arithmetic: real-with-margin` says a margin absorbs the rounding; it does not say *which* rounding, and on JAX that is not a property of the program (§2.3). Belongs beside `solver_options` for identical reasons and by the identical argument: **one jaxpr, three devices, three numerics** |
| `vacuity` | `witnessed` (a point in R was exhibited) \| `unchecked` — link 1, partially |
| `obligations` | Assumptions discharged elsewhere, if any — notably the induction base case (§10.8) |
| `coverage_ref` | The coverage artifact for this query. **A verdict without its scope is a verdict without its meaning** (§6.3) |

**Deliberately absent: source locations.** `SOUNDNESS.md`'s "cache the proof,
not the report" rule means the stamp is keyed on `query_hash`, which excludes
`source_info` by design. File and line pointers are re-derived from the current
jaxpr at render time and never stored in or restored from a cache. The first
violation of this reports a line number from someone else's file.

### 2.5 Engine disagreement is a first-class output

Design commitment 1 says the fuzzer, the SMT encoder, and the abstract
interpreter are three interpreters of one object. Read as an evidence
architecture rather than a design convenience, that is **three independent
implementations of one specification** — an N-version diversity argument handed
over for free, and the strongest **error-detection** mechanism stelling has
before certificates exist.

***Detection, and not the TD argument, and the two are not the same thing.***
This read *"the strongest TD (error-detection) argument stelling has"*, which
smuggles a confidence level out of a mechanism that produces information only
when it disagrees with itself. N-version diversity earns its reputation from
*disagreements observed*; three implementations that have never disagreed have
told you nothing, and the classical objection to N-version arguments — correlated
faults from a shared specification — bites hardest here, because these three
share a specification **by construction** (commitment 1: they are three
interpreters of *one object*). So the lattice is the strongest detector, its
firings are decisive, and what it supports in §12.2 is worked out there and is
not a level.

The engines are constrained by soundness into a lattice. **The constraints depend
on whether the chain contains a ⊤**, and that dependency is the whole subtlety:

| Engine | May produce | May never produce | Why |
|---|---|---|---|
| **Fuzzer / sampler** | `falsified` (checked witness), `unknown` | `verified`, `vacuous` | It is a search. Absence of evidence is not evidence — which bars it from `verified` *and* from concluding R = ∅ from a failure to find a point in R |
| **Abstract interpreter** (over-approximating) | `verified`, `vacuous` (⊥ reached), `unknown` | `falsified` | A spurious abstract counterexample is not a counterexample — unless it checks out concretely, in which case it is the fuzzer's result, not the AI's |
| **SMT, ⊤-free chain** (exact encoding) | `verified`, `falsified` (model → checked), `vacuous` (R unsat), `unknown` | — | Exact encoding: models are real, UNSAT is real |
| **SMT, chain containing ⊤** | `verified`, `unknown` | **`falsified` without a checked witness** | **This is the dual of the ⊤ rule** (§4.2). ⊤ over-approximates, so the encoded R′ ⊇ R and B′ ⊒ B: UNSAT still proves the property (⊤ costs precision, not soundness, in the green direction) but a **model may be an artifact of the over-approximation**. Checking the witness (§2.2) is what recovers `falsified`, which is why it is a construction invariant and not a test |

The fourth row is why `vacuous` is a claim only two of the engines can make, and
why §10.2 downgrades a missing sampler witness to `VACUOUS` rather than promoting
it: **failing to find a point in R is not the same as R being empty**, and only
SMT (R unsatisfiable) or AI (⊥) can tell the difference.

From the lattice, a mechanically detectable *contradiction* falls out:

> **One engine says `verified` while another produces a witness that checks out
> ⇒ something in the chain is broken, and here is a reproducer.**

**But it does not follow that stelling is at fault, and the tool must not say
so.** The two claims are about different objects: the checked witness is a
*float* fact about the *compiled* program, while `verified` is (under
`arithmetic: real-with-margin`, `trust_boundary: jaxpr`) a claim about the
*traced* jaxpr over *ℝ*. Both can be true at once. The candidates are:

| Candidate | Which link | Whose |
|---|---|---|
| Encoder or query-construction bug | 3 | **stelling's** |
| Unsound transfer despite its tier | 4 | **stelling's** |
| Solver wrong on this query | 5 | Shared (§11.2) |
| **The margin did not absorb the rounding** | 6 | **The user's — and this is a finding about their program, not a bug in stelling** |
| **XLA does not implement the jaxpr here** | 7 | The user's, and a much bigger deal than a stelling bug |

The last two are not failure modes of the disagreement mechanism — **they are the
most valuable output it can produce.** A query that is provable over ℝ and false
in float is precisely the thing commitment 2 defers and §2.3 link 6 says nobody
checks. A mechanism that surfaces one and then blames itself has thrown away the
finding. §2.2's own argument forbids the misattribution: the oracle sits on the
far side of *every* stelling trust boundary, so a disagreement with it cannot be
localised to stelling without further work.

So the output is **"the chain broke here; these are the candidates; here is the
reproducer"** — loud, distinct from both `verified` and `falsified`, and
ledger-triggering *only* once triage lands on links 3, 4, or 5 (§7). Discharging
the triage is manual and always will be; the mechanism's job is to find the
disagreement and to refuse to guess at its cause. No verifier the author is aware
of has this as a first-class output.

**The architectural consequence** — a real change from `design/founding.md`,
offered for adoption rather than as a restatement:

> **The fuzzer should run on every verdict, not only on `unknown`/timeout.**
> *Built, and not yet adopted: the probe exists as an opt-in keyword and runs on
> no verdict unless a caller asks for it (§2.3.1).*

`design/founding.md` Stage 1 scopes the fuzzer as a *fallback* on
unknown/timeout. Running it on a `verified` verdict costs the same and is not a
fallback at all — it is a **contradiction hunt with a real chance of catching an
encoder bug, an unsound transfer, a solver bug, or the ℝ/float gap biting** (links
3–6), using the only oracle over real behaviour there is. It is the SMT-side
analogue of what Stage 2's differential testing does for the
abstract-interpretation side, and the founding document's own reasoning for
differential testing applies verbatim: *"replay only exercises paths where a
violation was already found."* Fuzz-on-verified exercises the paths where one was
claimed impossible.

**What a contradiction hunt is, and what it is not.** Everything in the
paragraph above survives the correction in §2.3.1 because none of it claims the
hunt *finds* anything: a mechanism with *"a real chance of catching"* a defect is
described honestly by that phrase and by no stronger one. The moment it does
catch something the result is decisive — a `verified` and a checked violating
point cannot both stand, and §2.5's whole apparatus exists to route that. **The
moment it catches nothing, nothing has happened.** The budget argument below had
to be rewritten because it did not hold that line.

Budget: unbounded in CI over the corpus; a small time budget by default in
interactive use; `--no-crosscheck` to opt out, never silently.

***This read "unbounded in CI over the corpus (where it is the cheapest soundness
evidence available)", and the parenthesis is the error the capability table two
paragraphs up forbids.*** **A search that finds nothing is not cheap evidence; it
is not evidence.** The budget is worth spending for the opposite reason, and
§6.1 already argues it correctly: this is the only mechanism pointed at links 2
and 7 before Phase 5, a firing is decisive, and **the cost of a null result is
the whole point of paying it** — you are buying the chance of a refutation, not
a quantity of assurance. A budget defended as "cheap extra assurance" is a
budget that gets cut the first time CI is slow, which is exactly what §6.1 says
in as many words; a budget defended as the compilation gap's only detector is
defended correctly. **And it must be spent to buy anything at all:** as built,
the probe is `check(..., falsify="sample")` and **default-off**, and no CI job in
this repository turns it on, so the corpus budget described here is a plan and
not a running cost (§2.3.1).

### 2.6 The verdict artifact

**The content hash is the spine of the whole evidence architecture.** It is the
capability a component library structurally cannot have: a *stable, semantic
identity for the thing that was checked*, independent of file, line, and tracer
identity (`ir.py`, canonical up to alpha-renaming). It makes four things
possible that are otherwise impossible:

1. verdicts are content-addressed, so they can be *stored*, *compared*, and
   *found again*;
2. the proof cache can be keyed on the query rather than on the file
   (`SOUNDNESS.md`);
3. "did both solvers see the same query?" is a checkable question, which is what
   makes §2.5's cross-check meaningful rather than approximate;
4. **the audit query is decidable** (§7.3) — retroactive invalidation is only
   real because verdicts have identities that a scope predicate can range over.

#### The hash is not a total identity, and the architecture must not assume it is

`design/transparent-primitives.md` records the exception, and this document
called that sentence *"the single highest-value sentence in the repository's
documentation"* (§5.2) before nearly walking into it:

> **Hash caveat:** `OpaqueParam` content is invisible to the content hash — two
> programs identical except for their custom derivative rules hash alike.
> Irrelevant for forward analyses; **Stage-2 gradient/equivalence work must not
> key caches on the primal's hash alone.**

So `query_hash` equality means *"the same program, up to the contents of any
`OpaqueParam`"*. For every forward analysis — Stage 0 through Stage 2's
inductive invariants — that is exactly the identity relation verification wants,
and points 1–4 stand unqualified. For gradient properties, `custom_vjp`
equivalence, and anything else whose semantics depend on a thunk's contents, it
is **not an identity at all**: two queries that differ precisely in the thing
under test hash alike.

Three consequences, all cheap now and expensive later:

- **The verdict artifact carries `opaque_params`** — the count and the
  `(primitive, param)` slots the query contains — so that a hash match plus an
  empty list is a *total* identity, and a hash match with a non-empty list is a
  match anyone can see needs more.
- **§16.3's Replay mode requires `opaque_params: []`**, or the stronger identity
  a Stage-2 gradient hash provides once it exists. Citing a stored verdict on a
  bare hash match is the forbidden keying, in the one place where it would look
  most reasonable.
- **The ledger's `affects` predicate may range over `opaque_params`** (§7.2), so
  a Stage-2 soundness event can scope itself to exactly the queries the hash
  cannot distinguish.

**`SOUNDNESS.md`'s cache rule is a different rule and does not cover this one.**
"Cache the proof, not the report" is about *source locations* — the hash
deliberately excludes them, so a rendered verdict must re-derive file and line
from the current jaxpr. It makes no claim that a hash match implies semantic
identity. Nothing in the repository does. This subsection is where that gap is
recorded rather than assumed away.

The artifact is JSON, schema-versioned, and written under `evidence/` by CI and
by users who want a record:

```json
{
  "schema_version": "1",
  "verdict": "verified",
  "soundness": "proof",
  "query_hash": "sha256:9f2c…",
  "opaque_params": [],
  "harness": "positivity",
  "environment": {
    "stelling": "0.2.1",
    "jax": "0.10.2",
    "solver": "cvc5",
    "solver_version": "1.3.4",
    "transport": "wheel",
    "solver_options": {"nl-cov": "false", "nl-ext": "full", "…": "…"}
  },
  "arithmetic": "real-with-margin",
  "trust_boundary": "jaxpr",
  "vacuity": "witnessed",
  "tiers": [
    {"primitive": "add", "domain": "smt", "tier": "exact"},
    {"primitive": "reduce_min", "domain": "smt", "tier": "sound"}
  ],
  "obligations": [],
  "coverage_ref": "evidence/coverage/0.2.1/positivity.json",
  "probe": {
    "ran": true,
    "budget_s": 30,
    "points_executed": 1188,
    "points_admissible": 964,
    "obligations_unprobed": 0,
    "declined": {"no-exact-reading-of-this-program": 3},
    "licence": "work done, not a finding: this probe can only refute, it did not fire, and a null result is a fact about the sampler and not about the verdict"
  }
}
```

**Rationale.** Rendering is a view; this is the record. Everything an assessor
needs to ask "is this verdict still valid?" — and everything `stelling audit`
needs to answer it — is here, and nothing that would rot (source locations,
timings, machine names) is.

#### The probe field, and why it is not `"crosscheck": {"fuzzer": "no-counterexample"}`

**That is what this block said, and it was the worst sentence in this document**,
because of where it stood rather than what it claimed. §2.5's capability table
bars the sampler from `verified` on the ground that **absence of evidence is not
evidence**; this block then made *"the sampler did not find one"* a **retained,
content-addressed field of the record that this same section calls the spine of
the whole evidence architecture** — durable, quotable out of context, and sitting
in a list of fields every one of which is either a fact the verdict depends on or
a named limitation of it. A null search result is neither. The shipped module,
the banned-field test over `ProbeReport`, the banned-word test over its stamp
line and `preconditions.py`'s comment at the call site were all built to stop
exactly this, and the design of record specified it anyway.

**The position taken here, and it is the second of the two available.** The
field is **retained**, and it records **only what it can** — work done and work
declined — **with its licence written into the field itself**. It is not
deleted, and the three reasons are worth stating because the first instinct is
to delete it:

- **A finding is not the reachable value.** On an artifact whose `verdict` is
  `verified`, the probe cannot have fired: had it fired there would be no
  `verified` to write the artifact about. So a field holding the *finding* has
  exactly one reachable value and distinguishes nothing. A field holding the
  *work* has many, and they are the ones a reader actually needs.
- **The skip rate is part of the result**, and deleting the field would make
  "probed hard" and "declined everything" print the same — which is the failure
  mode the module's own second heading names. A probe that declined every
  obligation on a program its exact reading could not reach has done nothing, and
  a record that cannot say so lets *"the probe ran"* be claimed with nothing
  behind it.
- **What is retained is falsifiable, and `"no-counterexample"` was not.** A count
  of points executed can be wrong, and someone can go and check it.
  *"No counterexample"* cannot be checked, cannot be wrong, and cannot be
  re-derived; it is the shape of a claim that survives every audit by asserting
  nothing.

**The licence string is constant, and that is the argument for putting it in the
record rather than in the schema document.** Fields are quoted out of their
artifacts. A reader who lifts this one lifts the disclaimer with it, which is the
same reason `ProbeReport.stamp_line` carries its disclaimer in the same sentence
as its counts rather than in a note beside them. The name `crosscheck` is retired
**as a field name** — it named an adjudication between engines (§2.5's lattice),
and a sampler that found nothing has adjudicated nothing. The `--no-crosscheck`
flag and `tests/crosscheck/` keep the word, because there the word names a run
and a lattice rather than a stored result.

**What this field must never grow**, stated so the next schema version has to
argue with it rather than around it: a `passed`/`clean`/`ok` boolean; a
`confidence`, `score` or `coverage` number; a count of *points that did not
violate* presented as an achievement; or any aggregation across verdicts that
turns many null results into one summary. The first three are the exact strings
the shipped `ProbeReport` refuses to grow a field matching, and a schema that
grew them would have moved the defect from the module to the record.

**One caveat on this whole block.** It is a schema sketch and parts of it are
still plan rather than tree: `arithmetic: real-with-margin` names a dial position
the source tree does not have, and the `Verdict` type in §10.2 carries no probe
field at all. §2.3.1 says which of these fields exist today; **the artifact is the
record and may be wider than the type, but it may not be wider than the truth.**

---

## 3. Documentation structure

### 3.1 The tiers

Three tiers is the standard shape for a scientific library (API reference /
narrative guide / executable examples). A verifier needs a fourth, and the
fourth is the one that matters:

| Tier | Content | Audience | Rots if |
|---|---|---|---|
| **1 — API reference** | Autodoc over the public surface | Users writing harnesses | The API changes |
| **2 — Semantics** | What the query object is; what each transfer function and encoding *means*; what the arithmetic is | Users who need to know what a verdict claims; reviewers; assessors | jax changes, a transfer changes, a tier changes |
| **3 — Harnesses** | Executable, self-contained harnesses; the corpus | Users learning; the differential-testing bed | Real code moves |
| **4 — Evidence** | The soundness ledger; coverage; corpus results per release; the qualification package | Assessors; users deciding whether to rely on a verdict; future maintainers | **Never — it is retained, not maintained** (§6.2) |

Tier 4 is where a verifier's documentation architecture actually differs from a
library's. It is not documentation *about* the software; it is the software's
output, kept. It is append-only, versioned, and **must never be edited to look
better in retrospect** — §7.1.

### 3.2 The tree

**THE ANNOTATIONS BELOW ARE `EXISTS`/`NEW` AGAINST THE DAY THIS SECTION
WAS WRITTEN, AND TWO OF THEM DESCRIBE A MECHANISM THAT WAS NEVER BUILT.**
`evidence/` does not exist in the repository and never has; the
`SOUNDNESS.md` Log is written by hand. The `SOUNDNESS.md` line said *"Log
rendered from evidence/soundness.yaml"* in the present tense, 570 lines
above the §7 STATUS block that retracts it and 830 above §8.3's, until
2026-08-21. Read §7 and §8.3 for what is in force; read this tree as the
shape, not as an inventory of what shipped.

```
stelling/
├── README.md                      # positioning + the §1.3 rule + one green proof, one red
├── SOUNDNESS.md                   # EXISTS — verdict trust policy (normative); the LEDGER (§8.3); Log HAND-AUTHORED (§7 STATUS)
├── SECURITY.md                    # EXISTS — supply chain, private reporting
├── CONTRIBUTING.md                # EXISTS — extend with §9 checklists
├── CHANGELOG.md                   # NEW — with Soundness / Verdicts / Coverage sections (§8.3)
├── CITATION.cff                   # EXISTS — citation + configuration-management artifact
├── DOCUMENTATION_ARCHITECTURE.md  # this document
│
├── design/                        # EXISTS — normative design notes, dated, evidence-pinned
│   ├── founding.md                #   commitments + roadmap
│   ├── transparent-primitives.md  #   the prototype for §5.2
│   ├── value-model.md             #   the claim, the experiment, the pre-registered falsifier
│   └── _template.md               # NEW — status / evidence / re-verify trigger header
│
├── docs/
│   ├── user_guide/
│   │   ├── install.md
│   │   ├── harness_authoring.md   # incl. positive controls, base-case obligations (§10.8)
│   │   ├── reading_a_verdict.md   # what each field means and what it does not claim
│   │   ├── reading_coverage.md
│   │   └── ci_integration.md      # exit-code discipline (§10.9), evidence retention
│   │
│   ├── semantics/                 # TIER 2 CORE
│   │   ├── index.md
│   │   ├── query.md               # the (B, R) object — the one-sentence specification
│   │   ├── arithmetic.md          # ℝ-with-margin vs float; δ; what link 6 costs
│   │   ├── transfers/             # one doc per (primitive, domain) family — §5.3
│   │   │   ├── _template.md
│   │   │   └── …
│   │   └── backends/              # one doc per backend — §5.4
│   │       ├── _template.md
│   │       ├── z3.md
│   │       └── cvc5.md            # incl. the wheel/binary and option findings already in README
│   │
│   ├── evidence/                  # TIER 4 (prose face)
│   │   ├── index.md               # what stelling's evidence does and does not show
│   │   ├── trust_product.md       # §2.3, maintained as the canonical copy
│   │   ├── self_verification.md   # §6.1
│   │   ├── corpus.md
│   │   └── coverage.md
│   │
│   ├── qualification/
│   │   ├── positioning.md         # tool, not component; the §1.3 rule
│   │   ├── constraints_of_use.md  # the T2 artifact — §4.2
│   │   ├── trust_boundary.md      # jaxpr, not HLO
│   │   ├── package.md             # the per-release package — §11.1
│   │   ├── iec61508.md
│   │   ├── iso26262.md
│   │   ├── do330.md
│   │   └── en50128.md
│   │
│   ├── api/                       # TIER 1 (generated)
│   └── bibliography.bib
│
├── evidence/                      # TIER 4 (machine-readable, RETAINED, append-only)
│   ├── soundness.yaml             # the ledger — §7.2
│   ├── schema/
│   │   ├── verdict.schema.json
│   │   ├── coverage.schema.json
│   │   └── soundness.schema.json
│   ├── coverage/<version>/…
│   ├── corpus/<version>.json      # per-release corpus results — never deleted (§6.2)
│   └── releases/<version>/package.md
│
├── corpus/                        # TIER 3 — traced programs from real projects
│   ├── manifest.yaml              # source, commit, licence, what it exercises
│   └── …
│
├── examples/                      # TIER 3 — teaching harnesses
│
└── tests/
    ├── differential/              # concrete runs land inside computed bounds (§6.1)
    ├── replay/                    # every reported counterexample reproduces
    ├── crosscheck/                # engine agreement lattice (§2.5)
    └── …
```

**Tooling.** Sphinx + MyST for the eventual site, matching the ecosystem norm;
`sphinxcontrib-bibtex` for the bibliography. Deferred until there is a user base
(Appendix B, Phase 5) — the tree above is readable on GitHub as-is, and premature
site infrastructure is the classic way to spend a month not building a verifier.

**One rule about the split between `design/` and `docs/semantics/`:**
`design/` records *decisions and their evidence* at a point in time
(`transparent-primitives.md` is exemplary: status, the jax version it was
verified against, and the forcing function for re-verification).
`docs/semantics/` records *what is true now*. When they disagree, the design note
is history and the semantics doc is the claim. Neither is deleted.

---

## 4. Boundary and positioning documents

### 4.1 `docs/qualification/positioning.md`

The load-bearing statement, to be reproduced in `README.md` in short form:

> stelling is a verification tool for JAX array programs. It emits no code, links
> into no product, and executes nothing at runtime. Its output is a **verdict**:
> a claim, with a stamp naming everything the claim depends on.
>
> stelling is not a certified tool, and it is not qualified for any safety
> integrity level, ASIL, or software level. Tool qualification is a property of a
> tool *plus a use plus an argument*; no tool holds it in isolation, and any
> claim to the contrary is a category error.
>
> In the vocabulary of the functional-safety standards, stelling is a **class T2
> tool** (IEC 61508-3 §7.4.4, EN 50128 §6.7): errors in it **can fail to reveal
> defects but cannot directly create errors in the executable software**. Under
> DO-178C §12.2 it is criterion 3, or criterion 2 if its output is used to
> justify eliminating or reducing another process. Under ISO 26262-8 §11 its tool
> impact is TI1 or TI2 depending entirely on your use.
>
> **If stelling's output does not cause you to do less of anything else, stelling
> is not in your safety argument and needs no qualification. The moment a green
> verdict removes work, you own a qualification argument — and
> `docs/qualification/package.md` is what supports it.**
>
> A falsified verdict is a different object from a verified one. Its witness
> replays against your compiled function; you can check it in one line without
> trusting stelling at all. Nothing in this document's cautions applies to it.

**Input-trust boundary.** stelling assumes a trusted harness and a trusted
environment. It executes user-supplied Python during tracing and replay, by
construction — that is what tracing *is* — and it invokes a solver on user-derived
formulas. It performs no sandboxing and is not a security boundary. Running
stelling on a program is exactly as trusted an act as running that program.

### 4.2 `docs/qualification/constraints_of_use.md` — the T2 artifact

IEC 61508-3 §7.4.4 expects, for a T2 tool, documentation *"clearly defining the
behaviour of the tool and any instructions or constraints on its use."* This
document is that, and stelling can do something no other tool in Appendix A does:

> **The coverage report is a per-verdict, machine-generated constraints-of-use
> statement.** Every other tool ships a static "known limitations" chapter that
> is out of date by construction and that nobody reads at the moment of use.
> stelling emits, *with each verdict*, exactly which primitives fell to ⊤ and
> where precision died — for that query, on that day, at that version.

The static document therefore carries only what genuinely cannot be per-run:

| Constraint | Source |
|---|---|
| Fixed shapes per harness — no shape generalisation | Commitment 3 |
| Verdicts are about the **traced** program; XLA is outside the boundary | Commitment 5; link 7 |
| **The precision must be pinned, and pinned with a contract rather than a preference.** A verdict about an f32 `dot_general` is a verdict about a computation the jaxpr does not determine: `jax_default_matmul_precision` defaults to `None`, and `lax.Precision` is device-dependent by its own docstring — `DEFAULT` is bf16 on TPU and tf32-or-f32 depending on which GPU you have. **Use `lax.DotAlgorithmPreset` (e.g. `F32_F32_F32`), not `precision=`**: the preset raises at compile time on a device that cannot honour it, where the preference silently gives you something else. State the algorithm and the device class the verdict is asserted for; both go in the stamp (§2.4) | Link 7; verified against jax 0.10.2 / 0.11.0, 2026-07-16 |
| **Verdicts are asserted for a device class, not for JAX in general.** A verdict earned under `F32_F32_F32` on CPU says nothing about the same harness on a TPU that will run it in bf16 | Link 7 |
| Real arithmetic with margin — a proof over ℝ is not a proof over float until link 6 closes | Commitment 2; link 6 |
| Unknown primitives default to ⊤. ⊤ is a *sound* over-approximation, so a `verified` verdict whose chain contains a ⊤ is still a proof — you proved the property while knowing nothing about that value. **⊤ costs precision, not soundness**, and is the usual reason a verdict is `unknown` rather than a reason to distrust one that is not | Commitment 4 |
| A heuristic transfer in the chain demotes the verdict from `proof` to `heuristic` | Commitment 5; §5.1 |
| The solver is a separate, unqualified program identified by name, version, transport, and full option set | §11.2 |
| Vacuity is guarded only for R = ∅; **the harness author is responsible for the property being non-trivial** | §10.8 |
| Induction proves the step; **the base case is the user's obligation** unless discharged in the harness | §10.8 |
| **The jax series is not pinned.** `TESTED_JAX_SERIES` names the series stelling is *tested* against; the packaging floor is `jax>=0.5`, and `README.md` is explicit that it is "a documented tested-version floor, **not a binding**". Outside the tested series, transcription emits a `RuntimeWarning` and proceeds. **A verdict produced under an untested jax series is unverified territory that stelling warned about and permitted** — the warning is in the log, the jax version is in the stamp, and the choice was the user's | `_optional.py`, `_jax_compat.py`, `README.md` |

The vacuity, base-case, and jax-series rows are the ones that bite, and they are
use errors, not tool errors — which is precisely why they belong in a
constraints-of-use document rather than the soundness ledger. **The two precision
rows are a different animal and deserve a note**: they are the only constraints
here that are *actionable in the user's code* rather than merely cautionary.
Pinning an algorithm is a thing you do, once, and then link 7 is materially
narrower for every verdict thereafter. It is the highest-value line in this
document per unit of effort, and §13.1 argues it is also exactly the artifact a
third-party-software review will ask for.

### 4.3 `docs/qualification/trust_boundary.md`

Links 7 and 8, in full, **and without the flattering precedent.** Earlier drafts
of this document — and the sentence originates here, not in `founding.md` — said
that Kani makes the same bet on rustc's backend that stelling makes on XLA. It is
not the same bet, and the document that says so is the document that will get the
constraint of use wrong.

The contents, in the order that makes the argument:

1. **What Kani's bet actually is.** MIR → machine code, through LLVM. LLVM does
   not reassociate floating-point without a fast-math flag, and rustc does not set
   one. The backend has a documented contract and honours it, so the gap Kani
   carries is *"LLVM implements what it says it implements"* — a claim about
   correctness against a fixed specification.
2. **Why stelling's is a different shape.** stelling's gap is not only *"XLA
   implements what it says"* — it is that **for the float path there is no fixed
   specification to implement.** The jaxpr does not denote a unique numerical
   program. The evidence (verified against jax 0.10.2 and 0.11.0 on 2026-07-16,
   and re-verify on every series bump per `TESTED_JAX_SERIES`):

   | Fact | Source |
   |---|---|
   | `jax_default_matmul_precision` is `None` — unset, platform's choice | `jax.config`, both series |
   | *"The **device-dependent** `precision` argument … Has no impact on CPU backends."* | `lax.Precision.__doc__` |
   | `DEFAULT`: TPU → *"performs float32 computations in bfloat16"*; GPU → *"tensorfloat32 if available (e.g. on A100 and H100 GPUs), otherwise standard float32 (e.g. on V100 GPUs)"* | `lax.Precision.__doc__` |
   | `HIGH`: TPU → *"3 bfloat16 passes"*; GPU → *"tensorfloat32 where available, otherwise float32"* | `lax.Precision.__doc__` |
   | `Precision('bfloat16_3x')` and `Precision('tensorfloat32')` **both return `HIGH`** — one request, two names, different numerics per device | probed, both series |
   | `DotAlgorithm`: *"using an unsupported algorithm will raise a Python exception when the computation is compiled"*; `DotAlgorithmPreset.F32_F32_F32` exists | `lax.DotAlgorithm.__doc__`, both series |

3. **The line, and it is the link-7 analogue of `SOUNDNESS.md`'s line about solver
   options:** *"cvc5 1.3.4 said unsat" is not a reproducible claim.* Neither is
   *"the invariant holds at f32."* **One jaxpr, three devices, three numerics.**
4. **What commitment 2 buys and what it does not.** Robust invariants with slack
   are insensitive to reassociation — *"fusion shuffles ulps, robust invariants
   don't care"*. They are **not** insensitive to a precision *selection*: bf16
   against f32 is fifteen mantissa bits, and no margin sized for f32 rounding
   absorbs that. Nor does slack help against a miscompilation, which is not a
   rounding difference. The coupling between links 6 and 7 covers reassociation and
   stops.
5. **The partial defence, which is why this is a constraints document and not a
   lament.** Pin the algorithm (`DotAlgorithmPreset`), not the preference
   (`Precision`) — the first fails loudly where the second degrades silently. §4.2.
6. **Link 8, and why the chain does not stop at 7.** Translation validation would
   prove the emitted code implements the jaxpr; it proves nothing about the machine
   executing it correctly. §2.3.
7. **What translation validation to HLO would buy** — and what it would not: it
   closes 7 and leaves 8, and 8 has no closure.

### 4.4 What every document repeats

Three sentences appear in `README.md`, `positioning.md`, the package, and the
verdict renderer, and are allowed to be repetitive because a reader arrives at
exactly one of them:

1. A green verdict is a claim whose stamp names what it depends on.
2. A red verdict is a fact you can check yourself in one line.
3. What you eliminate determines your burden.

---

## 5. Semantics documentation standard

This is the analogue of a per-algorithm guide, and it is where a verifier's
documentation earns its keep. For a physics library, an undocumented assumption
produces a wrong number that someone might notice. **For a verifier, an
unjustified soundness claim produces a false proof — and the entire point of the
tool is that you stop looking after one.**

### 5.1 The tier is the claim

Commitment 5 gives three tiers — `design/founding.md`'s *"Transfer tiers.
Exact / sound / heuristic declared per transfer function"* — and this document
does not get to amend a design commitment. ⊤ is reported as `sound` with
`precision: none` and a `defaulted: true` flag rather than as a tier of its
own.

**THE TREE'S THREE ARE NOT THIS THREE, AND THIS SECTION FORECLOSED THE
QUESTION WRONGLY.** `stelling.propagate` declares `TIER_EXACT = "exact"`,
`TIER_SOUND = "sound"` and `TIER_SOUND_LIBM = "sound-libm"` — cited by symbol
and not by line, for the reason `SOUNDNESS.md`'s SF-0.2.0-14 gives — and
`propagate`'s own module docstring enumerates those three. `heuristic`
appears nowhere in `src/stelling/*.py`. Measured over the live registries:

```
real-mode transfers : exact 35, sound 13, sound-libm 2
ieee-mode transfers : exact 46, sound  2, sound-libm 2
```

So the tier set a verdict actually stamps in `transfer_tiers` is
`{exact, sound, sound-libm}`, and the third name in the table below is a tier
nothing has ever been assigned. **Whether `sound-libm` is a fourth tier or a
refinement of `sound` is a real question and this document is not the place it
gets settled** — the sentence "so there are three" read as though it had been,
which is the one thing a subordinate document must not do. What is not in
doubt: `sound-libm` carries an explicit named assumption
(`interval.EXP_LIBM_ASSUMPTION`) that plain `sound` does not, so the weakest-link
rule below has three inputs in the code and four names between the two pages.
The table's `heuristic` row is the commitment's vocabulary and is kept as such;
it describes no transfer in the tree.

| Tier | Means | The bar for claiming it |
|---|---|---|
| `exact` | The transfer computes precisely the concrete semantics; no approximation | A statement of the jax semantics being mirrored, and the test that pins it |
| `sound` | The transfer over-approximates: it never excludes a concrete behaviour | **A written argument.** Not a test. Not a citation. An argument, in the doc, that a reviewer can follow and attack |
| `heuristic` | May exclude concrete behaviours; a verdict through it is not a proof | A statement of *how* it can be wrong, and why it is worth shipping anyway |

Three rules:

1. **`sound` without an argument is the most dangerous artifact in the
   repository** and is grounds for refusing a contribution outright. It is worse
   than `heuristic`, which is honest, and worse than ⊤, which is loud. Review
   weight follows: a `sound` tier claim gets more scrutiny than the code that
   implements it, because the code has tests and the claim has only the
   argument.
2. **Aggregation is by weakest link.** A verdict's `soundness` field is `proof`
   only if every transfer in the chain is `exact` or `sound`. One `heuristic`
   demotes the whole verdict. Rendering a heuristic-chain verdict as a bare
   `VERIFIED` is a use error the tool must not permit (§2.1 rule 3).
3. **The tier is data, not prose.** It lives in `TransferMeta` (§10.1), is
   surfaced in the coverage report and the verdict, and the doc's tier field is
   generated from the code — never the reverse. A tier that exists only in a
   Markdown table is a tier that will drift.

### 5.2 Primitive semantics are evidence, not lore

`design/transparent-primitives.md` is the prototype for this entire tier, and it
should be read as the template it is. What it does right, and what every
semantics document must therefore do:

| It does | Why it matters |
|---|---|
| States a **status** and that it is **normative** for a named scope | A reader knows whether they may rely on it |
| Names the **jax version the evidence was verified against** (0.10.2) and the **date** | jax churns; undated claims about jax semantics become lies silently |
| Names the **forcing function for re-verification** (`TESTED_JAX_SERIES`) | Re-verification that depends on someone remembering does not happen |
| Distinguishes **observed** from **expected** (`closed_call`/`custom_lin` "not observed on 0.10 but expected in other series") | An assessor can tell which claims are load-bearing |
| Records what is **deliberately unregistered** (`custom_lin`'s `bwd`) and that **first contact raises loudly** | The absence is a decision, not an oversight |
| Records a **consequence that will otherwise be inherited silently** (`OpaqueParam` is hash-invisible ⇒ Stage-2 gradient caches must not key on the primal hash alone) | This is the single highest-value sentence in the repository's documentation, because it is a landmine defused years before it is stepped on |

**Every claim about what jax does must carry the version it was verified
against.** This is the discipline that makes link 2 (transcription) credible over
time, and it is cheap only if it is habitual.

### 5.3 The transfer function document template

One document per `(primitive, domain)` family. **No invented identifiers**: the
natural key is `primitive@domain` (`gather@smt`, `reduce_min@interval`), and the
repository already has a content hash for identity where identity is needed.
Inventing an ID scheme for things that have natural keys is ceremony, and this
project does not do ceremony.

```markdown
---
bibliography: ../../bibliography.bib
---

# `<primitive>@<domain>`

**Module**: `stelling.transfers.<domain>`
**Tier**: exact | sound | heuristic     <!-- generated from TransferMeta -->
**Stability**: experimental | provisional | stable | deprecated
**jax semantics verified against**: jax <X.Y.Z>, <date>

## What the primitive does

[The concrete semantics being modelled, as *verified*, not as remembered.
State how it was verified — a traced example, a pinned test, the jax
documentation plus a confirming trace. Cite the jax version. If the primitive
lowers to something surprising, say so here and say when you checked.
`jnp.roll` traces to a `jit` equation on jax 0.10 — that class of fact belongs
in this section, with its date.]

## The transfer function

[The mathematical definition. LaTeX. The domain's abstract semantics, the
concretisation, and the transfer.]

$$
\widehat{\text{add}}(\hat{a}, \hat{b}) = [\underline{a} + \underline{b},\ \overline{a} + \overline{b}]
$$

## Tier argument

**Tier: sound.**

[**Mandatory for `sound` and `heuristic`.** For `sound`: the argument that
γ(transfer(â, b̂)) ⊇ {a + b | a ∈ γ(â), b ∈ γ(b̂)}. Written so a reviewer can
attack it. Cite the rounding treatment explicitly — outward rounding, nextafter
widening, MPFR — because a sound transfer implemented in unsoundly-rounded
arithmetic is unsound (`design/founding.md`, Stage 2, soundness plumbing).

For `heuristic`: the specific way it can be wrong, and the reason it ships.

For `exact`: a sentence naming the concrete semantics it mirrors, and the
pinning test.]

## Precision

[Where it loses, and to what. The interval domain cannot represent the energy
ellipse; say so *here*, in the transfer's own document, not only in a roadmap
bullet. This section is the raw material of the coverage report's
"where precision died" line.]

## Encoding map

| Term | Implementation | Notes |
|---|---|---|
| $\underline{a} + \underline{b}$ | `stelling.transfers.interval.add` | outward-rounded via `nextafter` |
| ⊤ fallback | `stelling.transfers.registry.top` | reported as `defaulted: true` |

[Every term in the transfer definition appears here. A term handled directly by
a backend construct (a z3 expression constructor, a solver option) is documented
as such rather than silently omitted. CI checks every qualified name resolves —
§5.5.]

## Differential evidence

- Test: `tests/differential/test_<primitive>_<domain>.py`
- Corpus entries exercising it: [list or query]
- Concrete runs executed against the computed bounds: [count, per release]
- **Runs landing OUTSIDE the bounds: [count — and any number above zero is a
  soundness event (§7), not a line item]**
- **The scope this figure is a null over**: which shapes, dtypes and ranges were
  drawn, and which were not

*A count of runs that landed inside the bounds is a null result, and this
template used to record only that. It licenses nothing about the transfer (§6.1,
§2.3.1); what it records is that a search ran and how big it was, which is worth
retaining and is worth nothing without its scope beside it. A transfer document
that reports the first figure and not the last two is reporting how tired the
searcher got.*

## Known imprecision and failure modes

[Numbered. Feeds `constraints_of_use.md` and the coverage report.]

## References

[Pandoc-style `[@Key]`; keys resolve against `docs/bibliography.bib`; CI checks.]

## Changelog

| Version | Date | Change | Verdict-flipping? |
|---|---|---|---|
| 0.2.0 | 2026-… | Initial | n/a |
```

**The `Verdict-flipping?` column is not decoration.** Per `SOUNDNESS.md`, any
change that flips a verdict is a soundness event regardless of the semver bump —
including a *precision improvement* (§8.2). The column is where the author
notices that before the ledger has to.

### 5.4 The backend document template

One per backend (`z3`, `cvc5`, later dReal, later CROWN-style bound propagation).
Contents:

| Section | Why |
|---|---|
| **Transport** | Wheel vs. external binary; what `--show-config` reports; what the wheel lacks. The README's cvc5 findings move here and are maintained here |
| **The emitted option set** | The full set, with the reason for each. `SOUNDNESS.md` forbids invoking a solver on defaults; **this document is where each option's presence is justified**, including options whose emitted value currently coincides with the default |
| **Encoding rules** | jaxpr construct → solver term. The exact/sound/heuristic tier of each |
| **Known solver behaviour** | Version-pinned observations. The `exp`/`nl-cov` finding in `SOUNDNESS.md` is the exemplar: three configs, three engines, one version string |
| **Fragment** | Which theories, and what falls outside |
| **What a verdict from this backend does not claim** | Link 5, per backend |

### 5.5 CI bridges — documentation rot prevention

Documentation that can rot silently is worse than none in a tool whose product is
trust. Four checks, all cheap, all standalone scripts (no Sphinx dependency):

| Check | Script | Fails when |
|---|---|---|
| **Tier agreement** | `scripts/check_tiers.py` | A transfer doc's `Tier:` header disagrees with `TransferMeta.tier` in code. **Code wins**; the doc header is generated |
| **Encoding map resolves** | `scripts/check_encoding_map.py` | A qualified name in an Encoding map column does not resolve via `importlib` + `getattr` to a callable |
| **Citations resolve** | `scripts/check_citations.py` | A `[@Key]` has no entry in `docs/bibliography.bib`. Unused entries warn; `_`-prefixed templates are excluded |
| **Coverage of the registry** | `scripts/check_transfer_docs.py` | A transfer is registered in code with no document, or a document names no registered transfer |

The fourth is the one with no analogue in a library's documentation: **a transfer
function that exists in code and not in the semantics tier is an undocumented
soundness claim**, and CI should refuse it.

---

## 6. Evidence architecture: verifying the verifier

### 6.1 What each technique catches — and cannot

**The jitted function is the only oracle over the program's behaviour.** Most
techniques below are applications of it, and those are the only ones that can
tell you what the code actually does — which is why the corpus is load-bearing
rather than decorative. The three rows with a different oracle are marked, and
their limits follow from what they are checking instead.

| Technique | Oracle | Catches | Structurally cannot catch | Link | Phase |
|---|---|---|---|---|---|
| **Witness checking** (§2.2) | Execute: assumes on the witness, then the witness under `jit` | Encoder bugs on paths where a violation was found; models that are artifacts of ⊤ | Anything on a verified path — *"replay only exercises paths where a violation was already found"* (`founding.md`) | 3 | 1 |
| **Raise-on-unknown-param** | The transcriber's whitelist | A param type with no transcription rule — refuses to guess | A faithful transcription of a *misunderstood* primitive; anything about primitives it does recognise | 2 | done |
| **Per-primitive pinned traces** | Execute `make_jaxpr`; compare | Transcription that drops or mangles a covered primitive; jax lowering changes on the pinned series | **Any primitive not pinned.** Note what does *not* belong here: `test_ir.py`'s round-trip is IR → dict → IR and never touches jax, so it checks *serialization*, not transcription. A transcriber that dropped an eqn would round-trip perfectly | 2 | 2 |
| **Positive controls** (§10.8) | The harness's own mutation | Harnesses that cannot fail — B ≡ true | A harness that is merely *weaker* than intended | 1 | 1 |
| **Fuzz-on-verified** (§2.5) | Execute sampled points under `jit` — **the compiled program, on the real device** | A `verified` that is false, from **any** cause below the harness: a mis-transcription (2), a wrong query (3), an unsound transfer (4), a solver bug (5), the ℝ/float gap biting (6), XLA's precision selection (7), even a faulty machine (8) | A false `verified` whose counterexamples are sparse in R; **and it cannot say which link broke** (§2.5). **A run that finds nothing licenses nothing** — the same cut as the cross-solver row below | **2–8** | 1 |
| **Corpus breadth** | Real code | Primitives, lowerings, and shapes the author did not imagine | Anything the corpus does not contain | 2, 4 | 2 |
| **Differential testing** | Execute corpus programs concretely | Transfer functions whose bounds exclude real behaviour | Imprecision (bounds too wide is not a bug); regions the corpus does not reach | 4 | 3 |
| **Cross-solver agreement** | **None** — the solvers only check each other | One solver wrong where the other is right; encoding divergence | Both wrong the same way (shared theory bugs are not hypothetical). **Agreement is not evidence** — it is the absence of one kind of disagreement | 5 | 3 |
| **Cross-engine lattice** (§2.5) | Witness checking adjudicates | Contradictions between the three interpreters of one query | *Which* link broke — that is triage, and it is manual. Joint agreement that is jointly wrong | 3–6 | 3 |
| **Proof certificate + checker** | **The proof, not the program** | Solver unsoundness (link 5), completely | Everything else. A checked proof of the wrong formula is still a checked proof — links 1, 3, 4, 6, 7 are untouched | 5 | 5 |

(Phases are Appendix B's, which hang off `design/founding.md`'s stages.)

Two entries deserve emphasis because they are stelling's, not inherited:

- **Fuzz-on-verified is worth far more than a fallback, and the reason is its
  oracle.** It executes sampled points under `jit` — the compiled program, at
  whatever precision XLA selected, on the actual silicon. That is ground truth for
  **every link from transcription down**, because the thing it runs is the thing
  that ships. Concretely: a transcriber that quietly analysed a different program
  (link 2) is caught by running the real one; XLA's precision selection (link 7) is
  not modelled, it is simply *present*, because the sampled point ran through it.

  So the honest scoping is stronger than "the cheapest contradiction detector
  before certificates," which is how earlier drafts of this document sold it:

  > **Fuzz-on-verified is the only mechanism pointed at links 2 and 7 at all,
  > ever, before Phase 5's translation validation.** Nothing else in this
  > architecture executes the compiled artifact against a green claim. Differential
  > testing runs concrete programs but checks transfer bounds (link 4). Cross-solver
  > has no oracle. Certificates check the proof, not the program.

  That changes what the sampling budget buys. It is not insurance against a solver
  bug that probably isn't there — it is **the compilation gap's only detector**, and
  the compilation gap is the one this document spends §4.3 and §2.3's first reading
  establishing is worse for JAX than for anything Kani faces. A budget argued for on
  those grounds is argued for correctly; one argued as "cheap extra assurance"
  invites being cut first. Run it on the accelerator *and* on CPU (§2.3, link 8) and
  it is a differential against 7 as well as a check of it.

  ***"Detector", not "defence", and the word was changed rather than softened.***
  A detector that has not detected anything has not defended anything, and the
  block quote above is careful to say *pointed at* where the sentence under it
  said *defence* — two spellings of one quantity, one screen apart, in the
  paragraph whose whole subject is scoping this mechanism honestly. Link 7 has a
  detector and no defence; the correct disclosure is that link 7 is bare, and
  §2.3.1 makes it against the tree. Nothing else in this bullet changes: the reach
  argument is right, the budget argument is right, and the CPU differential is
  still worth its one line.
- **The cross-engine lattice** requires no extra implementation: commitment 1
  already mandates three interpreters of one object. The only work is to *compare
  them, and to treat a disagreement as a fault to be located rather than as a
  verdict to be reported* — §2.5 is emphatic that the location is not knowable
  from the disagreement alone.

### 6.2 The corpus is a record you cannot backfill

`design/founding.md` scopes the corpus three ways: the primitive census picks the
registry order; the corpus is the differential-testing bed; and *"later, the
credibility artifact."* The third use has a requirement the first two do not, and
it is the reason this section exists:

> **"History of successful use" is a history. It is a record kept over time, and
> a release that shipped without one can never acquire one.**

ISO 26262-8's *increased confidence from use* (method 1a) — the only qualification
method a solo open-source project can support directly (§15) — asks for
documented evidence of the tool version, the use cases, the period of use, and
the malfunctions observed. IEC 61508-3 §7.4.4.4 accepts, as part of the evidence
that a T2 tool conforms to its documentation, *"a suitable combination of history
of successful use in similar environments and for similar applications."*

Neither accepts a green CI badge. Both accept a retained, versioned record.

**The architectural requirement is therefore on CI, not on the corpus**:

1. Every release runs the full corpus and **writes `evidence/corpus/<version>.json`
   into the repository** — verdicts, tiers, coverage, timings, environment.
2. That file is **never deleted and never edited**. A later release does not
   correct an earlier record; it adds a new one, and a soundness event (§7)
   explains any divergence.
3. Every corpus entry carries provenance in `corpus/manifest.yaml`: upstream
   project, commit, licence, what it exercises, why it is in the corpus.
4. **Malfunctions observed are part of the record, and the record must say so
   even when there are none.** Both frameworks ask for observed malfunctions as a
   component of a usage history; the ledger (§7) is where stelling's live. What
   makes the record credible is not that it has entries — it is that the *search*
   is documented: which techniques ran, over what corpus, at what budget, **over
   what scope**, finding what. **A record that reports "no soundness events"
   alongside "the corpus, differential suite, and cross-checks that ran, over
   this scope, and found none" is a record. A record that reports only the first
   is not a record of anything.**

   *This said the first kind of record "is strong", and strong is a word about
   evidence. It is not: a documented search that found nothing is a documented
   search that found nothing, and its whole value is that a reader can see what
   was looked for and go and look somewhere else. The scope clause is added for
   the same reason — a search record without its scope cannot be read at all,
   which is `design/lessons-ledger.md`'s standing rule that an instrument's
   silence is a reading only if the instrument could have spoken.*

Point 4 inverts the instinct, and it needs care in two directions.
`SOUNDNESS.md`'s log is currently `*(empty — no releases yet)*`, and an empty log
is **not** in itself a liability: a soundness event is a *verdict flip*
(`SOUNDNESS.md`), so the first release has no prior verdicts to flip and
structurally cannot have one, and an honest release that changes no verdicts
cannot either. The rule is therefore not "the log must fill up" — that is
pressure to log a non-event, which corrupts the artifact this section exists to
protect. The rule is: **whatever the log says, the record of what was looked for
must be next to it.** An empty log with no search behind it says nothing; an
empty log next to a documented search says something worth reading. Nobody here
speaks for how a given assessor reads either — Appendix D is explicit that
stelling has no relationship with any assessor, ever.

### 6.3 The coverage report is the scope statement

`design/founding.md` calls the coverage report "most of the UX" — it turns
"unknown" from an insult into a work item. In the qualification frame it is
something more: **it is the verdict's scope statement, generated rather than
written**, and it is the artifact that makes an honest green verdict possible at
all.

| Field | Why an assessor cares |
|---|---|
| Primitives that fell to ⊤, with source locations | The verdict says nothing about these paths |
| Where precision died, and which transfer | Distinguishes "not provable" from "not provable *by this domain*" |
| Tier of every transfer in the chain | Whether the verdict is a proof (§5.1) |
| Defaulted vs. defined transfers | Work item vs. deliberate over-approximation |
| The fragment the query landed in (QF_LRA, QF_NRA, …) | Whether the solver was in a decidable fragment |
| Vacuity witness | Whether R was inhabited |

It is emitted per verdict, retained per release, schema-versioned, and
machine-readable. **A green verdict whose coverage report is not retained
alongside it is not evidence** — it is an assertion.

### 6.4 What stelling's evidence does not show

`docs/evidence/index.md` must state this as plainly as `SOUNDNESS.md` states its
policy. What stelling's evidence provides:

- **Verdicts**, with their assumptions named.
- **Counterexamples**, replayed against the compiled function — facts about the
  program, established by execution.
- **Coverage** — a machine-generated scope statement per verdict.
- **A soundness ledger**, with retroactive invalidation that is mechanically
  queryable.
- **A usage record**, retained per release.

What it does not provide, and cannot:

- **Not a claim about the compiled program**, until link 7 closes. A verdict is
  about the traced jaxpr.
- **Not a claim about float execution**, until link 6 closes. A verdict under
  `arithmetic: real-with-margin` is a claim about ℝ, with margin, and the
  margin-absorption argument connecting it to IEEE-754 does not exist yet.
- **Not a claim that the harness is the right property.** Link 1 is the user's.
  stelling checks R ≠ ∅ and nothing else about intent.
- **Not a solver qualification.** stelling identifies the solver exactly; it does
  not vouch for it (§11.2).
- **Not a tool qualification.** §1.2.
- **Not completeness.** `unknown` is common, expected, and reported. A tool that
  returned `verified` more often would be a worse tool, not a better one.

---

## 7. The soundness ledger

**STATUS, because this section is written in the present tense and one of
its clauses is not true yet.** `evidence/soundness.yaml`, its generated
JSON face, and the CI job that renders the `SOUNDNESS.md` Log from them
**do not exist** — there is no `evidence/` directory in the repository,
and the Log is hand-authored. Everything below is the design for a
machine-queryable face, kept because the argument for it is the reason
`SOUNDNESS.md` is written the way it is; nothing below describes a
shipped mechanism. **The ledger in force is `SOUNDNESS.md` itself**, and
§8.3 is the clause that says so. The one sentence to read as a promise
rather than a report is *"`SOUNDNESS.md`'s Log section is rendered from
the ledger by CI"*: today it is written by hand, and the drift that
guards against is guarded by review.

### 7.1 The ledger is the architecture, not a document

`SOUNDNESS.md` is the strongest artifact in the repository, and the reason is one
sentence: *"which prior verdicts are retroactively invalid."* No other verifier
in Appendix A promises that. It is a **recall mechanism for evidence**, and it
addresses the failure mode that actually destroys trust in verification tooling:
a tool ships a soundness fix, every prior claim is now suspect, nobody knows
which ones, so nobody re-runs anything, so everyone quietly keeps the old
beliefs. The alternative that organisations actually adopt — *never upgrade the
verifier* — is worse and is why so many qualified toolchains are frozen a decade
behind.

Retroactive invalidation is only real if it is **mechanical**. A prose sentence
saying "verdicts involving nonlinear real arithmetic between 0.3.0 and 0.4.2 may
be affected" cannot be evaluated against ten thousand stored verdicts, and so it
will not be. That gap — between a promise made in prose and a promise that can be
executed — is what this section closes.

**The division of labour with `SOUNDNESS.md`:**

- `SOUNDNESS.md`'s **Policy** section is hand-written, normative, and stays prose.
- `SOUNDNESS.md`'s **Log** section is **rendered from the ledger** by CI, so the
  human-readable log and the machine-queryable ledger cannot drift.
- **Entries are hand-authored.** This is deliberate and is the sign-off act:
  *writing the scope predicate is the judgement*, and it cannot be automated.
  Automation renders; it never decides.

#### The ledger has two faces, and the reason is the zero-dependency commitment

`pyproject.toml` declares `dependencies = []`; `README.md` promises **zero
required dependencies**; `_optional.py` states the rule — *"Everything outside
the standard library is therefore declared as an extra… never by a module-level
import."* YAML is not in the standard library. A ledger that only exists as YAML
would make PyYAML stelling's first hard runtime dependency, in the service of a
module (§10.7, §16.1) whose entire value proposition is that it imports with
nothing installed.

So the ledger is two files, and CI keeps them in step:

| File | Form | Who reads it | Why |
|---|---|---|---|
| `evidence/soundness.yaml` | Hand-authored YAML | Humans; the maintainer writing an entry | Comments, block scalars, and readable predicates. This is the **source of truth**, and authoring it is the sign-off act |
| `evidence/soundness.json` | **Generated** | `stelling audit`; downstream tooling; assessors' scripts | `json` is stdlib. `stelling.evidence` reads this and imports with **zero dependencies** |

CI renders the JSON and the `SOUNDNESS.md` Log from the YAML on every push and
fails if either is stale. Authoring stays pleasant; consumption stays free.
Contributors touch only the YAML; **the generated files are committed** — an
assessor cloning the repository at a tag must not need a build step to read the
ledger, and `stelling audit` must not need one to run.

The same split applies to any evidence artifact a human authors. Everything
stelling *emits* — verdicts, coverage, corpus records (§2.6, §6.3) — is JSON
already and needs no second face.

### 7.2 The event record and the scope predicate

```yaml
# evidence/soundness.yaml
schema_version: "1"
generated: "2026-…"

events:
  - id: STEL-SND-0001
    title: "reduce_min@interval ignored NaN operands, yielding a finite lower bound"
    class: unsoundness          # unsoundness | incompleteness | precision | crash | verdict-stability
    discovered: "2026-…"
    discovered_by: differential-testing   # how it surfaced — part of the usage record
    description: >
      The interval transfer for reduce_min took the min of lower bounds
      without propagating NaN, so a query over an array that could contain
      NaN could be reported verified when a concrete NaN made B false.

    affects:
      stelling: ">=0.3.0,<0.4.3"
      verdict: [verified]
      transfers_include: ["reduce_min@interval"]
      # optional further narrowing:
      # arithmetic: [real-with-margin]
      # solver: {name: cvc5, version: ">=1.3,<1.4"}
      # opaque_params_nonempty: true   # Stage-2: the hash cannot distinguish these (§2.6)
      # query_hash_in: [...]           # when the blast radius is enumerable

    invalid: true               # verdicts matching `affects` are retroactively invalid
    fixed_in: "0.4.3"
    resolution: >
      Re-run any matching verdict under >=0.4.3. Verdicts that were
      falsified or unknown are unaffected.
    ledger_note: >
      Found by the corpus differential suite before any user report.
      No falsified verdict is affected: witness checking is unaffected by
      this defect.
```

`fixed_in` and `resolution` are the only two fields an existing entry may ever
acquire or change (§7.4). Everything else is written once.

**`affects` is a predicate over verdict stamps, not a paragraph.** Every key
corresponds to a field the stamp carries (§2.4), which is the entire reason the
stamp must be structural rather than conventional. The predicate is evaluable
against a stored `verdict.json` with no human in the loop.

**On classes.** `SOUNDNESS.md`'s policy is deliberately broader than unsoundness:
*any* change that flips *any* verdict is a soundness event. That breadth is
correct and is preserved — the log is about **verdict stability**, not only
soundness. The `class` field classifies *within* the log rather than gatekeeping
entry into it:

| Class | Means | Retroactively invalidates? |
|---|---|---|
| `unsoundness` | A verdict was wrong in the dangerous direction (false `verified`) | **Yes** |
| `incompleteness` | A verdict was `unknown` that should have been decidable | No — nothing was over-claimed |
| `precision` | A transfer improved; verdicts move `unknown` → `verified` | No — but it *is* a verdict flip and is logged (§8.2) |
| `verdict-stability` | Same query, different verdict, for any other reason (option change, solver bump) | Case by case; the predicate says |
| `crash` | Refused to run where it previously ran | No |

Only `unsoundness` sets `invalid: true` by default. But **all five are logged**,
because the promise in `SOUNDNESS.md` is that a user can ask "did anything about
my verdict change?" and get a complete answer — and because §6.2 point 4 means
the log's contents are the malfunction half of the usage record.

### 7.3 The audit query

```
$ stelling audit evidence/verdicts/ --ledger evidence/soundness.yaml

  312 verdicts examined  (stelling 0.3.0–0.4.2, jax 0.10.2)

  INVALID (3)   STEL-SND-0001  reduce_min@interval / NaN
                  advection-positivity      verified  0.3.1  → re-run
                  halo-bounds               verified  0.3.4  → re-run
                  neighbor-list-inbounds    verified  0.4.0  → re-run

  REVIEW  (1)   STEL-SND-0004  gradient/custom_vjp equivalence unsoundness
                  vjp-agreement             verified  0.4.1  → predicate needs
                                                              opaque_params;
                                                              stamp predates the
                                                              field (added 0.4.2)

  STALE   (2)   query hash no longer matches the current trace
                  cfl-bound                 verified  0.4.2  → harness changed
                  leapfrog-energy           verified  0.4.2  → harness changed

  OK    (306)
```

Three findings, three distinct meanings:

- **INVALID** — a ledger predicate matched. The verdict is withdrawn. Re-run.
- **REVIEW** — a predicate could not be decided because the stamp lacks a field
  the predicate needs. Note what this cannot be, if §2.4 is honoured: it cannot
  be a *missing* stamp field that existed at the time, because
  `SOUNDNESS.md` requires the full option set and Phase 0 makes the stamp
  structural from v0.1.0. The realistic cause is **schema growth** — a field
  added in 0.4.2 that a 0.4.1 stamp could not have carried, however
  conscientious 0.4.1 was. That is legitimate and unavoidable, which is why
  REVIEW is a *finding* and not an accusation. It becomes a bill for a shortcut
  only when the field did exist and was skipped; the ledger entry's author can
  tell which, and the audit cannot.
- **STALE** — the harness no longer traces to the stored query hash. The verdict
  is about a program that no longer exists. **This is the mechanical answer to
  the harness-drift use error** (§10.8) and it comes free from content
  addressing. Bear §2.6's caveat: a hash *match* is identity only up to
  `OpaqueParam` contents, so STALE is sound (a changed hash always means a
  changed program) while its absence is not (an unchanged hash may still be a
  changed program, if `opaque_params` is non-empty). **The audit reports
  absence-of-STALE with a non-empty `opaque_params` as REVIEW, not OK.**

`stelling audit` is the single most valuable command the tool will ship for
anyone with a pipeline, and it is implementable the day stamps and the ledger
exist — long before the interesting verification work is done.

### 7.4 Lifecycle and gates

Two phases, one of which must not be automated:

**Phase 1 — Discovery.** A GitHub issue, labelled `soundness`. Reproduction,
root cause, the argument about scope. This is engineering discussion and belongs
in the open.

**Phase 2 — Formalisation.** A hand-authored entry in `evidence/soundness.yaml`,
with a predicate. **The judgement is in the predicate**: too narrow and invalid
verdicts survive; too broad and users re-run work for nothing and stop believing
the ledger. This is the sign-off, and it is exactly why it stays manual.

CI enforces the mechanics, not the judgement:

| Check | Gate |
|---|---|
| `evidence/soundness.yaml` validates against its schema | Every push, blocking |
| Every `affects` key names a real stamp field | Every push, blocking |
| No entry deleted or mutated except `fixed_in` / `resolution` | Every push, blocking — **history is append-only** |
| `SOUNDNESS.md`'s Log and `evidence/soundness.json` match the render of the YAML | Every push, blocking (§7.1) |
| Every issue labelled `soundness` and closed has an entry | **Release gate, no grace period** |
| The release's corpus record is written to `evidence/corpus/<version>.json` | Release gate, blocking |

**One tier, no grace period.** A component library can reasonably grade its
anomaly formalisation by severity — a cosmetic defect need not block a release.
A verifier cannot: `SOUNDNESS.md` already says *"a soundness fix that ships
without a log entry is itself a soundness event."* The grace period is zero
because the artifact being protected is the one thing the tool sells. The
compensating design is that the ledger is *narrow* — it admits only verdict
flips, not every bug — so the gate is cheap to satisfy. Ordinary bugs live in
GitHub issues and the changelog like anywhere else.

---

## 8. Versioning, stability, and what a pin buys

### 8.1 SemVer governs the API, not verdicts

Already policy (`SOUNDNESS.md`). The architectural consequences:

| A user asks | The answer |
|---|---|
| "Is 0.4.3 a safe upgrade from 0.4.2?" | For the **API**, semver answers. For **verdicts**, semver says nothing — run `stelling audit` against the ledger |
| "Will my verdicts still hold?" | Only the ledger knows. That is what it is for |
| "Can I pin and stop thinking?" | You can pin. You cannot stop thinking, because the solver, jax, and the transport are not pinned by pinning stelling (§8.4) |

### 8.2 Precision improvements are verdict flips

This follows directly from `SOUNDNESS.md`'s policy and is counterintuitive
enough to state explicitly, because the instinct is to treat it as a pure win:

> **Making a transfer function more precise flips verdicts from `unknown` to
> `verified`. That is a verdict flip. It is a soundness event and it gets a log
> entry.**

It does not invalidate anything (`class: precision`, `invalid: false`), so the
cost is one YAML entry. The benefit is that the ledger remains a *complete*
answer to "did anything about my verdict change?" — and completeness is the only
property that makes an audit tool worth running. A ledger that records only the
scary changes is a ledger you must supplement with your own investigation, which
is a ledger nobody uses.

The same logic applies in the other direction, and this one does bite:
**weakening or removing a transfer function flips `verified` → `unknown`.** That
is a coverage regression, it is a verdict flip, and a user who upgrades and
suddenly cannot prove what they proved last month deserves to find the reason in
the log rather than in their own debugging.

### 8.3 `CHANGELOG.md`

Keep a Changelog, plus three sections that exist because this is a verifier:

```markdown
## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

### Soundness
- [Every entry added to SOUNDNESS.md this release, with its ID. Links to
  the ledger; never restates the predicate — one source of truth.]

### Verdicts
- [Every known verdict flip not already covered above: new transfers, tier
  changes, default option changes, solver version bumps that moved anything on
  the corpus.]

### Coverage
- [Primitives that left ⊤. Precision improvements. Fragments newly reachable.
  This is the section users will actually read, because it answers "can I
  prove more than last month?"]
```

**`SOUNDNESS.md` IS THE LEDGER, and the Soundness section links to it.**
This clause named `evidence/soundness.yaml` until 0.2.0, and the rule
above drifted for one reason: **that file was never built.** There is no
`evidence/` directory in the repository, so §8.3 routed the detail to a
file that does not exist and the detail landed in the two files that do —
by 0.2.0 the changelog's Soundness section was **2990 of `CHANGELOG.md`'s
3778 lines, 79.1% of it**, against a released 0.1.0 section of 60. The
YAML ledger was a Day 1 artifact the project outgrew, and §7 is retained
as the design it was rather than rewritten into a description of
something shipped: read §7 as the plan for a machine-queryable face, and
this clause as the rule in force.

So, in force: `SOUNDNESS.md` is the official soundness ledger and the
single source of truth for a soundness entry's predicate, measurement,
scope and derivation. `CHANGELOG.md`'s Soundness section carries **strict
one-liners** — ID, one-sentence statement, affected versions, and a link
to that entry's section in `SOUNDNESS.md`. It never restates a predicate.

The 0.2.0 detail was ROUTED there rather than summarised away, and the
move is checked rather than asserted: `tests/test_soundness_routing.py`
pins every block by hash and requires the two files to partition the same
ID set in both directions, so an entry cannot be brought into compliance
by deleting it. A future release adds entries to `SOUNDNESS.md` and a
one-liner apiece to `CHANGELOG.md`; it does not need that manifest, which
records the moves that were made.

**THIS CLAUSE'S LETTER GOVERNS THE SOUNDNESS SECTION AND ITS RATIONALE IS
WIDER, so a second section was routed under the same machinery.**
`### The eager construction-site detector (Mode 2), DEFAULT-OFF` was 242 of
`CHANGELOG.md`'s 1158 lines — 20.9%, the third-largest section — and it
carried predicate, measurement and derivation exactly as the Soundness
section had. It is routed into `SOUNDNESS.md` under
**0.2.0 Mode 2 detail**, with its own manifest section, its own
`source_commit`, a derived span and per-block hashes, and the SAME
partition proof in both directions. The rule for a future section is the
rule this one followed: route it if the changelog is carrying the
predicate, and check the move rather than assert it. The rule is NOT that
every section becomes one-liners — `### Known limitations (0.2.0)` and the
release's `### Coverage` are what a reader opens the changelog for, and
they state no predicate the ledger owns.

### 8.4 The pin is not the environment

**A pinned stelling version does not pin a verdict.** The stamp has seven
environment fields (§10.2), and stelling controls two of them: its own version,
and the option set it emits — which is exactly why `SOUNDNESS.md` requires the
option set to be emitted in full and never left to a default. The other five move
under the user.

| Moves independently | Consequence |
|---|---|
| **jax** | Transcription changes; `jnp.roll`'s lowering is a fact about a jax series, not a constant. `TESTED_JAX_SERIES` is the honest claim, and a warning fires outside it |
| **The solver** | z3/cvc5 upgrade under the user's resolver. `SOUNDNESS.md`'s finding — same version, three option sets, three engines — means even a *pinned* solver is not a pinned engine |
| **The transport** | A wheel and an external binary of the same cvc5 version are different programs with different feature sets (`--show-config`) |
| **Solver defaults** | Which is why `SOUNDNESS.md` forbids invoking on defaults: a stamp that records "defaults" records what was *not* asked for |

So the reproducibility unit is not `stelling==0.4.2`. It is
**`(stelling, jax, solver, solver_version, transport, option_set, query_hash)`**
— the stamp, entire. Documentation that says "pin stelling for reproducibility"
is wrong and must not be written. §14 is the full treatment.

### 8.5 API stability levels

| Level | Meaning | Guarantee |
|---|---|---|
| `stable` | Covered by semver | Breaking changes only in major |
| `provisional` | May change in minor with a deprecation cycle | One minor's notice |
| `experimental` | May change without notice | None |
| `deprecated` | Scheduled for removal | Removed next major |

Applied to: the harness API (`any_array`, `any_pytree`, `assume`, `assert_`,
`nonvacuity`, `trace`), the verdict artifact schema, `stelling.ir`, and the
evidence schemas.

*This list read `any_array`, `any_scalar`, `assume`, `assert_`, and it was
wrong in both directions. It named `any_scalar`, which `design/founding.md`
sketched and nobody built — the string occurs in this file, in that one, and in
no Python anywhere — and it omitted `any_pytree` and `nonvacuity`, which ship.
This is the section that assigns semver stability levels, so a reader takes it
for an inventory of the surface rather than a plan, and an inventory that names
an absent function while missing two live ones is the wrong kind of wrong
here.*

*And the repair for that was still an entry short. `stelling.harness.__all__`
holds **six** names and the corrected list gave five: `trace` was missing,
although `harness.py`'s own module docstring documents `trace(harness)` and
says in as many words that "the six names below" are re-exported from
`_jax_compat`. Derived from `__all__` rather than retyped from memory this
time, which is what the standard above was asking for and did not get: an
inventory that misses one live function is a smaller instance of the same
wrong kind of wrong, and it was written by the sentence complaining about it.*

*The two further occurrences of `any_scalar`, in §10.8's positive-control
example, sit inside a sketch of a PROPOSED idiom — `@stelling.harness`,
`@stelling.control`, an `assume=` keyword — none of which exists either, and
which that section presents as a remedy to build. They are left alone: a sketch
may name what it proposes; it is this section, the stability inventory, that may
not.*

*That read "§9's". Both occurrences are under `### 10.8 Harness authoring
controls — the use-error class`; §9 is* Contributor standards *and merely
cross-references §10.8 for the positive-control rule, which is where the wrong
number came from. §15's renumbering check cannot reach this and neither can any
gate: §9 RESOLVES against a heading, it is just not the heading the sentence
means. A section reference that lands on the wrong section is the quieter half
of the defect §15 records, and counting is no use against it.*

**The schemas are the stable surface that matters.** A user's pipeline reads
`verdict.json` and `soundness.yaml`; an assessor reads them years later. They get
`stable` first and change slowest — before the Python API does, because a
schema break orphans retained evidence and retained evidence is the one thing
that cannot be regenerated (§6.2).

---

## 9. Contributor standards

`CONTRIBUTING.md` already carries the three ground rules that matter (SPDX, the
jax import boundary, the soundness-event obligation). §9 is what it grows into as
the transfer registry appears.

### 9.1 New transfer function

```markdown
### New transfer function checklist

- [ ] Registered in the transfer registry for exactly one `(primitive, domain)`,
      and the census constraints still hold at import (they raise, so a
      half-registered row breaks the package rather than shipping quietly)
- [ ] **Tier declared** — the second element of the registry entry — and it
      reaches the verdict: `transfers_used` stamps `(primitive, tier)`
- [ ] **For `sound` or `heuristic`, the ARGUMENT is written** beside the transfer,
      in its docstring, naming the jax version its semantics were verified
      against. A `sound` claim without an argument is not reviewable and will not
      be merged. **This is the bar; everything else here is mechanical**
- [ ] For `sound`: the rounding treatment is named. Outward rounding is not
      optional — a sound transfer in unsoundly-rounded arithmetic is unsound.
      A transfer that does no arithmetic says so, and is `exact`
- [ ] **Containment test in `tests/`**: draw a box, run the transfer, execute jax
      at concrete points inside it, and require every output element to land in
      the computed interval. **With both controls** — a positive one showing the
      harness reaches the row, and a negative one showing it reports a violation
      when the box is wrong (see `tests/test_dot_general_interval.py`)
- [ ] At least one corpus entry exercises it, or a note saying why none does
- [ ] Coverage report shows it as defined, not defaulted: `measure()` returns
      `unknown = 0` and `transparent = 0` on a query that uses it
- [ ] If it flips any recorded verdict, an entry in `docs/verdict-ledger.md`
- [ ] **State what each gauge you rely on REACHES** (Norm G). A gauge that drives
      the emission plans does not test the transfer, and vice versa — name the
      face, and do not quote "zero survivors" as coverage of a face it never ran
```

**Not yet infrastructure.** The following were specified for this checklist
before the repository had anywhere to put them, and are kept here as a roadmap
rather than as merge conditions. Requiring them today would mean a bar that
references six directories that do not exist — which reads to an outside
contributor as stale documentation, not as rigour.

| deferred item | what it needs first |
|---|---|
| `TransferMeta` attached | the type itself; tier currently rides in the registry tuple |
| tier argument in `docs/semantics/transfers/<primitive>@<domain>.md` | that tree; the argument is currently in the transfer's docstring, which is where it is actually read |
| differential test in `tests/differential/` | the directory; containment tests currently live in `tests/` |
| encoding map complete, CI resolves every term | the encoding map |
| changelog **Coverage** section entry | a `CHANGELOG.md` |
| bibliography entry | `docs/bibliography.bib` |

**Substance over location.** Every deferred row above is satisfied *somewhere* for
the rows shipped so far — the tier argument, the rounding statement, the
containment test and the corpus entry all exist and all execute. What is missing
is the filing system, and a filing system is not a soundness property. Move each
row up into the checklist as its infrastructure lands.

**The bar is the tier argument.** Everything else on the list is mechanical.

### 9.2 New backend

```markdown
### New backend checklist

- [ ] `docs/semantics/backends/<name>.md` per the §5.4 template
- [ ] The **full option set** is emitted explicitly, never defaults, and every
      option's presence is justified in the doc — including options whose value
      currently coincides with the default (`SOUNDNESS.md`)
- [ ] The stamp records name, version, transport, and the emitted option set
- [ ] Counterexample replay works: every model this backend produces converts to
      concrete arrays and reproduces under `jit`. **Replay is the encoder's unit
      test** (`design/founding.md`, Stage 0) — a backend without it is not done
- [ ] Cross-solver **disagreement** test against an existing backend on the
      corpus, keyed on `query_hash` so "same query" is a fact and not a hope —
      **a sat/unsat disagreement must fail loudly and must never be resolved by
      picking.** Passing means no disagreement was found on what was run; it is a
      merge condition, not evidence that the backend is right (§6.1)
- [ ] The engine's position in the §2.5 lattice is declared and enforced by test
- [ ] `python -m stelling` reports it, its transport, and its feature set
```

### 9.3 New harness / corpus entry

```markdown
### New corpus entry checklist

- [ ] `corpus/manifest.yaml`: upstream project, commit, licence, what it
      exercises, why it is here
- [ ] Licence is compatible and recorded. The corpus is redistributed; REUSE
      must stay green
- [ ] Traces on the pinned jax series; the primitive census is regenerated
- [ ] If it is a *teaching* harness (`examples/`) rather than corpus: it carries
      a positive control (§10.8) — a mutation that must come back falsified
```

### 9.4 Documentation standards

- **Every claim about jax semantics carries the version it was verified against
  and the date.** Non-negotiable; §5.2.
- **Every claim about solver behaviour carries version, transport, and options.**
  `SOUNDNESS.md`'s `exp`/`nl-cov` finding is why.
- **Docstrings are NumPy-style; mathematical variable names may match published
  formulas** (`tau`, `dt`, `dx`) rather than house style.
- **References go in `docs/bibliography.bib`**, cited Pandoc-style `[@Key]`, with
  a human-readable inline description alongside each key so documents read
  without Pandoc.
- **`design/` notes carry the `transparent-primitives.md` header**: status,
  scope of normativity, evidence with version and date, and the forcing function
  for re-verification.

---

## 10. Code-embedded hooks

The design philosophy differs from a component library's in one specific way, and
it follows from §2.4:

> A library's metadata is documentation *about* a computation, so it is optional
> and defaulted to `None`. **stelling's metadata is part of the claim, so it is
> mandatory and structural.** If it can be omitted, it will be, and the verdict
> that omits it is not a weaker verdict — it is a false one.

### 10.1 `Tier` and `TransferMeta`

```python
# src/stelling/transfers/meta.py  — no jax, no solver, stdlib only

from dataclasses import dataclass
from enum import Enum


class Tier(Enum):
    """Assumption tier of a transfer function (design commitment 5).

    Three tiers, because the commitment names three. ⊤ is SOUND with
    precision=None and defaulted=True — a fourth tier would amend a
    commitment, and this is not the place to do that.
    """
    EXACT = "exact"          # mirrors the concrete semantics
    SOUND = "sound"          # over-approximates; never excludes a behaviour
    HEURISTIC = "heuristic"  # may exclude behaviours; a verdict through it is not a proof


class Stability(Enum):
    EXPERIMENTAL = "experimental"
    PROVISIONAL = "provisional"
    STABLE = "stable"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class TransferMeta:
    """Structured metadata for one (primitive, domain) transfer function.

    Every field is required except `precision` and `defaulted`. This is
    deliberate: a transfer that does not know its own tier cannot be
    registered, because the tier is what the verdict inherits.
    """
    primitive: str                 # jax primitive name, e.g. "reduce_min"
    domain: str                    # "smt" | "interval" | "fuzz" | …
    tier: Tier
    stability: Stability
    jax_semantics_verified: str    # jax version the primitive's semantics were checked against
    verified_on: str               # ISO 8601 date of that check
    argument_ref: str              # docs/semantics/transfers/<primitive>@<domain>.md#tier-argument
    precision: str | None = None   # None means ⊤: no precision at all
    defaulted: bool = False        # True when this is the ⊤ fallback, not a written transfer

    def key(self) -> str:
        return f"{self.primitive}@{self.domain}"
```

**Registration refuses a `SOUND` or `HEURISTIC` tier whose `argument_ref` does
not resolve to a heading in a file that exists.** CI checks it; the registry
checks it at import in debug builds. This is the mechanism behind §5.1 rule 1 —
without it, the rule is a wish.

### 10.2 The stamp

```python
# src/stelling/verdict.py

@dataclass(frozen=True)
class Environment:
    stelling: str
    jax: str
    solver: str
    solver_version: str
    transport: str                       # "wheel" | "binary:/path" | "none"
    solver_options: tuple[tuple[str, str], ...]   # the FULL emitted set, sorted
    solver_features: tuple[str, ...]     # NO DEFAULT — see below

    def __post_init__(self) -> None:
        # SOUNDNESS.md: "for an external cvc5, its --show-config feature set".
        # A default of () would let a binary-transport Environment be built
        # without the one field that distinguishes two builds of the same
        # version number — which is the exact failure this type exists to
        # prevent, committed inside the type that prevents it.
        if self.transport.startswith("binary:") and not self.solver_features:
            raise ValueError(
                "external-binary transport requires the solver's feature set "
                "(--show-config); two builds of one version are two programs"
            )


@dataclass(frozen=True)
class Verdict:
    """The product. There is no constructor that omits the stamp.

    SOUNDNESS.md specifies the contents; this type is why they cannot be
    dropped. A Verdict without its environment is not an under-documented
    verdict — it is not a verdict.
    """
    result: Result                       # VERIFIED | FALSIFIED | VACUOUS | UNKNOWN
    query_hash: str
    opaque_params: tuple[str, ...]       # (primitive, param) slots the hash cannot see (§2.6)
    environment: Environment
    tiers: tuple[TransferMeta, ...]
    arithmetic: Arithmetic               # REAL_WITH_MARGIN | FLOAT_EXACT
    trust_boundary: TrustBoundary        # JAXPR | HLO
    vacuity: Vacuity                     # WITNESSED | UNCHECKED
    coverage_ref: str                    # the scope; a verdict without it has no meaning
    obligations: tuple[str, ...] = ()    # e.g. "base case x0 ∈ S not discharged here"
    witness: Witness | None = None       # FALSIFIED only; checked before construction

    @property
    def soundness(self) -> Soundness:
        """Weakest link. One heuristic transfer demotes the whole verdict."""
        if any(t.tier is Tier.HEURISTIC for t in self.tiers):
            return Soundness.HEURISTIC
        return Soundness.PROOF
```

Two invariants enforced at construction, not documented as guidance:

- **`FALSIFIED` requires a `witness`, and the witness is checked — both halves
  (§2.2) — before the `Verdict` exists.** Confirm it lies in R by evaluating the
  assumes concretely, *then* run the jitted function. An unchecked witness is
  precisely the claim §1.3 says is exempt, made without doing the thing that
  earns the exemption; a witness checked only against the assertion may be an
  artifact of ⊤ over-approximating R.
- **`VERIFIED` requires `vacuity is WITNESSED`.** Otherwise the result is
  `VACUOUS` or `UNKNOWN`. This is the Stage-0 guard, in the type.

### 10.3 The coverage artifact

Emitted with every verdict, schema-versioned, retained. §6.3 lists the fields.
The one API requirement: **the coverage report is not optional output.** A
verdict object carries a `coverage_ref`; a rendering that omits coverage is a
rendering that omits the scope.

### 10.4 Corpus and benchmark registration

```python
# src/stelling/evidence/corpus.py

@dataclass(frozen=True)
class CorpusEntry:
    key: str                  # natural key: "jax-cfd/advection/upwind"
    source: str               # upstream repo URL
    commit: str               # upstream commit — provenance, not a version range
    licence: str              # SPDX id; the corpus is redistributed
    exercises: tuple[str, ...]  # primitives / lowerings / fragments this pins
    rationale: str            # why it is in the corpus
    expected: Result | None = None   # None = "whatever it says, record it"
```

`expected` is usually `None`, and that is the point: the corpus is not a test
suite with a pass/fail. **It is a record of what stelling said about real code at
a version.** A corpus entry whose verdict changes between releases is not a
failure — it is a *finding*, and it goes to the ledger (§7) to be classified.
Wiring the corpus as ordinary pass/fail CI would suppress exactly the signal it
exists to produce.

### 10.5 Stability machinery

A `@stability(Stability.STABLE, since="0.3")` decorator with a registry and a
generated report. Straightforward, deferred to Phase 4, and a no-op identity
decorator until then so annotations can be written from day one without waiting
for the machinery.

### 10.6 `stelling audit`

```
stelling audit <verdicts-dir> [--ledger evidence/soundness.yaml]
                              [--retrace <module>]   # STALE detection
                              [--format json|text]
```

Exit codes matter more than the output format:

| Code | Meaning |
|---|---|
| 0 | All verdicts OK |
| 1 | At least one INVALID |
| 2 | At least one STALE |
| 3 | At least one REVIEW (a predicate could not be decided against the stamp) |

**3 is a real failure, not a warning.** It means an older stelling under-stamped
and the question cannot be answered. Treating it as a warning is how the answer
becomes "probably fine."

### 10.7 The ledger machinery

```python
# src/stelling/evidence/ledger.py — no jax, no solver, STDLIB ONLY (reads the
#   generated JSON face, never the YAML source — see §7.1)

def load(path: str) -> Ledger: ...

def matches(event: Event, verdict: dict) -> bool:
    """Evaluate an event's `affects` predicate against a stored verdict stamp.

    Raises Undecidable when the stamp lacks a field the predicate needs —
    which the audit reports as REVIEW rather than silently passing.
    """

def audit(verdicts: Iterable[dict], ledger: Ledger) -> AuditReport: ...
```

**Importable without jax, without a solver, and without anything else.** The same
reasoning that put `stelling.ir` behind the jax boundary applies here and matters
more: an assessor, a CI job, or a downstream project must be able to validate
evidence and run an audit without installing a JAX stack. Evidence tooling that
requires the tool's dependencies is evidence tooling that will not be run by the
people who most need to run it — and that argument does not survive making PyYAML
a hard dependency to reach it, which is why the ledger has a generated `json`
face and this module reads that (§7.1).

### 10.8 Harness authoring controls — the use-error class

A verifier's use errors are not misclicks. They are ways of believing something
the tool did not say. This is the analogue of a usability-engineering analysis,
and it is the section with the most new material relative to
`design/founding.md`.

| # | Use error | Guarded today? | Mechanism |
|---|---|---|---|
| 1 | **Empty region** — R = ∅, everything vacuously true | **Yes** | Sampler must exhibit a point in R; verdict is `VACUOUS`, never `VERIFIED` (Stage 0) |
| 2 | **Trivial property** — B ≡ true on R; the harness cannot fail | **No** | See below — this is a gap |
| 3 | **Reading `unknown` as fine** | Partly | Coverage report; **exit-code discipline** (§10.9) |
| 4 | **Harness drift** — you verified `step`, you ship `step_v2` | **No** | `stelling audit --retrace` → STALE (§7.3). Comes free from content addressing |
| 5 | **Over-reading the arithmetic** — "positivity proven" read as float positivity | Partly | `arithmetic` field in every verdict |
| 6 | **Over-reading the horizon** — inductive step read as run safety | **No** | See below — this is a gap |
| 7 | **Not pinning the environment** | **No** | §8.4; the stamp makes the omission visible after the fact |
| 8 | **Eliminating work on a green verdict without an argument** | N/A | §1.3, stated everywhere |

**Gap 2 — the vacuity guard is half a guard.** `design/founding.md`'s guard
checks R ≠ ∅. It does not check that B is non-trivial on R. A harness that
asserts something structurally true — `assert_(x == x)`, or, far more commonly,
an assertion whose failure the assumes have quietly excluded — passes the vacuity
guard and returns `VERIFIED`, truthfully and uselessly. **Vacuity has two failure
modes and stelling guards one.**

The remedy is the assay's, not the solver's: a **positive control**.

```python
@stelling.harness
def positivity():
    rho = stelling.any_array((N,), assume=lambda r: (r >= 0).all())
    dt  = stelling.any_scalar(bounds=(0.0, 10.0))
    stelling.assume(dt * u / dx <= 1.0)
    stelling.assert_((step(rho, dt) >= 0).all())

@stelling.control(of=positivity, expect=stelling.FALSIFIED)
def positivity_control():
    """Drop the CFL assume. If this does not go red, `positivity` proves nothing
    about CFL — the harness is not sensitive to the thing it claims to check."""
    rho = stelling.any_array((N,), assume=lambda r: (r >= 0).all())
    dt  = stelling.any_scalar(bounds=(0.0, 10.0))
    stelling.assert_((step(rho, dt) >= 0).all())
```

The control is a *harness authoring practice* supported by the tool, not tool
magic — the tool cannot know what the engineer meant, which is the whole reason
link 1 is the user's. What the tool can do:

- run controls in the same invocation and fail loudly if one does not go red;
- refuse, as a hard error, any query whose B is *syntactically* constant-true
  after construction — a cheap subset, worth having, catches the embarrassing
  cases;
- require, in the harness-authoring guide, that **every harness intended to
  license the removal of work ships with at least one control.** That is the
  narrowest and most defensible place to make a control mandatory, and it lines
  up exactly with §1.3.

**Gap 6 — induction has two halves and stelling proves one.**
`design/founding.md` commitment 6 states the inductive step: `x ∈ S ⟹ step(x) ∈
S`. The base case — `x₀ ∈ S` — is not mentioned, and an engineer reading
"inductive invariant proven" as "the 40-hour run stays in S" is reading a claim
that requires both halves. The step alone says nothing about a run that starts
outside S.

The remedy is structural, not documentary: the induction harness API **must**
either take the initial state and discharge the base case, or record it as an
explicit unfulfilled obligation in the verdict:

```python
Verdict(
    result=Result.VERIFIED,
    obligations=("base case: x0 ∈ S is not discharged by this harness",),
    ...
)
```

An `obligations` list that renders next to every verdict is how a proof of half a
thing stops being read as a proof of the whole thing. And it generalises: any
future assume-guarantee composition (`design/founding.md`, Medium) creates
exactly this shape — locally verified, neighbour assumed — and the assumed
neighbour is an obligation someone must discharge. **Building `obligations` into
the verdict now costs one field and prevents the entire class.**

### 10.9 Exit-code discipline

The failure mode of this tool is people reading absence of red as green. The exit
codes are therefore part of the safety design, not an ergonomic detail:

| Code | Result |
|---|---|
| 0 | All queries `VERIFIED` (or `FALSIFIED` where a control expected it) |
| 1 | Any `FALSIFIED` |
| 2 | Any `VACUOUS` |
| 3 | **Any `UNKNOWN`** |
| 4 | Any control failed to go red (§10.8) |
| 5 | **Engine disagreement — the chain broke** (§2.5) |

`--allow-unknown` exists, is opt-in, and is named so that it appears in the CI
config where a reviewer sees it. **`unknown` must never exit 0 by default**: a
tool whose default posture makes "I could not tell" indistinguishable from "I
proved it" has built the use error into its interface.

**Code 5 is the one code whose cause is unknown at the time it is emitted.**
Every other code is a verdict about the user's program; code 5 says two of
stelling's engines disagree about one query, which means a link broke — and,
per §2.5, the break may be stelling's (links 3, 4, 5) or it may be the
ℝ-versus-float or traced-versus-compiled gap biting the user's program (links 6,
7), which is a finding worth more than the verdict would have been. The message
must name the candidates and refuse to guess. A code 5 that announces "stelling
bug" is wrong roughly as often as it is right, and each time it is wrong it
discards the most interesting thing the tool ever found.

---

## 11. The qualification package

### 11.1 Structure

Published per release at `evidence/releases/<version>/package.md`. This is what a
user hands their assessor, or mines for their own tool-qualification argument.

```markdown
# stelling Tool Qualification Package — v[X.Y.Z]

## 1. Tool identification
Name, version, release date, licence, repository, PyPI, sdist/wheel SHA-256,
PEP 740 attestation URL, Python floor, jax series tested (TESTED_JAX_SERIES),
solver backends and versions tested, transports tested.

## 2. Tool classification
IEC 61508-3 / EN 50128: class T2 — errors can fail to reveal defects but
cannot directly create errors in the executable software. stelling emits no
code and touches no build artifact.
DO-178C §12.2: criterion 3, or criterion 2 if the verdict is used to justify
eliminating or reducing another process.
ISO 26262-8 §11: TI1 or TI2, determined by the user (§1.2, §1.3).
The determination is the user's. This section states the shape, not the class.

## 3. Functional specification
Given a boolean-output jaxpr B and a region R, decide whether
∃x ∈ R with B(x) = false, and return verified / falsified / vacuous / unknown
with a stamp naming every assumption the answer depends on.
That is the whole specification. [§11.1 note below.]

## 4. Constraints of use
docs/qualification/constraints_of_use.md, verbatim. Plus: the coverage report
accompanying each verdict is the per-query, machine-generated constraints
statement, and is the authoritative one.

## 5. The trust argument
The eight links (§2.3), in full, with the current state of each — including
link 8 (hardware), which no roadmap item closes and which is stated because a
chain ending at 7 claims to reach "the program that runs" and reaches "the
machine code that was emitted." This is the section an assessor will spend
their time in. It states plainly which links are unchecked at this version.

## 6. Known soundness events
Rendered from evidence/soundness.yaml at this version. Every event, its class,
its scope predicate, and whether it invalidates. Plus: how to run
`stelling audit` against your own retained verdicts.

## 7. Evidence, and what each item is evidence OF
**Read the two groups separately. They are not the same kind of claim.**

*Established facts — each is a positive result about a specific artifact:*
- Corpus: N entries, M primitives exercised, results at evidence/corpus/<v>.json
- Replay: every reported counterexample reproduced — count, and any that did not
- Tier arguments resolved: N of M registered transfers, with the unresolved named

*Searches conducted — each is a record of work done and of the scope it covered.
**None of these licenses a verdict; a firing of any of them refutes one:***
- Differential: concrete runs executed against computed bounds — count, coverage,
  **and the shapes/dtypes/ranges NOT drawn**; runs landing outside: count
- Cross-solver: queries where z3 and cvc5 disagreed, by query_hash; and how many
  ran with fewer than the full portfolio answering
- Cross-engine: lattice violations detected — count, **and the scope searched**
- Usage history: releases, dates, corpus results retained since v[…]
- Test suite: count, platforms, jax series

*This block was one undifferentiated list under the heading "Evidence", and four
of its seven rows were null results — a differential count, an agreement count, a
lattice count annotated "should be 0", and a test-suite count. **"Should be 0" is
the error in its purest form and it was in the artifact handed to an assessor.**
A zero there means the search did not fire; it is not a target and it is not a
score. The split above is the fix, and the second group is only worth retaining
because §6.2 is right that a record of "no events" is worth nothing without the
record of the search beside it — which makes these rows the *search record* and
not the evidence.*

## 8. Dependencies and the solver
§11.2. The solvers are separate, unqualified programs. Identification, not
endorsement.

## 9. Configuration management
§14. Git, tags, PyPI immutability, PEP 740 attestations, CITATION.cff,
the reproducibility unit (§8.4).

## 10. What this package does not claim
§6.4, verbatim.
```

**Section 3's brevity is a qualification asset, and it is narrower than it
looks.** IEC 61508-3 §7.4.4.4 asks for evidence that a T2 tool *conforms to its
specification **or documentation***, and the "or documentation" is where the rest
of stelling lives.

What the one sentence fixes is the **decision problem**: what stelling is asked,
what it may answer, and what an answer means. That is genuinely one sentence, and
it is genuinely unusual — the fuzzer, the SMT encoder, and the abstract
interpreter are not three features to specify separately, they are three
implementation strategies for one problem (commitment 1). A compiler's decision
problem cannot be stated in a sentence at all.

What the one sentence does **not** fix is the **semantics of B**. B is a jaxpr,
so its meaning is the meaning of every primitive it contains, at a particular jax
version — which is not one sentence, it is the whole of §5's semantics tier, and
it is why §5.2 insists that every claim about jax carries the version it was
verified against. **Requirements, not design**: an assessor asking what stelling
conforms to is owed both the decision problem and the primitive semantics, and
§15.2's "the transfer registry and its documents are the detailed design" would
paper over that if read alone.

So the honest claim is: **the specification is one sentence plus a semantics tier
that grows with the registry** — small at the top and unbounded at the bottom.
That is still far better than a manual, because the top is where conformance is
argued and the bottom is where it is *tested* (§5.5, §6.1). But commitment 1's
dividend is the decision problem's size, not the tool's, and overstating it in
the one document written for an assessor is the wrong place to be generous with
oneself.

Commitment 1 is therefore worth more in the qualification package than in the
codebase, and worth saying so: commitments erode under feature pressure, and this
one has a defender it did not know about.

### 11.2 The solver problem

**stelling cannot qualify z3, and it will not pretend to.** z3 and cvc5 are large
C++ systems with no qualification kits, no safety-standard development process,
and long histories of soundness bugs found by fuzzing campaigns — like every
solver, including the good ones. They are excellent software and they are
unqualified software, and both facts matter.

Link 5 is therefore the most exposed link in the green direction, and stelling's
honest options are exactly three:

| Option | What it buys | What it costs | Status |
|---|---|---|---|
| **Identify exactly** | Reproducibility. A verdict from "cvc5 1.3.4, wheel, options {…}, query `<hash>`" is a reproducible claim; "cvc5 1.3.4 said unsat" is not | Nothing | **Policy** (`SOUNDNESS.md`) — and policy only. The stamp that implements it is Phase 0; no code emits one today |
| **Cross-check** | An independent implementation disagreeing is a detected defect. Real diversity — z3 and cvc5 share almost no code | 2× solve time; catches nothing when both are wrong the same way, which shared-theory bugs are. **Agreement is not evidence** (§6.1) | Phase 3 |
| **Make trusting it unnecessary** | Link 5, and only link 5 (§11.3) | Substantial work; a new trusted component | Phase 5 |

Nothing else is available. "Increased confidence from use" for a solver — the
SMT-COMP record, decades of deployment — is a real argument and a user may make
it. It is not stelling's argument to make on their behalf, and the package should
say so and stop.

### 11.3 The certificate answer — and what it moves rather than removes

`design/founding.md` lists machine-checkable proof certificates under "Long —
hard and strategic," with the note *"required the moment this touches a regulated
pipeline."* The trust product (§2.3) sharpens that considerably:

> A certificate is not an auditor's convenience. **It is the only mechanism that
> can make a green verdict independently checkable at all** — the artifact that
> gives the universal claim what the existential claim already has.

And by the standards' own logic it does more than that. DO-178C §12.2.1 requires
qualification only when a tool's output is relied upon *without being verified*.
A checked witness is a verified output, which is why the red direction is exempt
(§1.3). **A checked proof certificate makes the green verdict's output verified
too — by the same clause, on the same reasoning.** The green direction joins the
red one in the bucket that clause exempts.

**The caveat is the entire point and must never be dropped:** the burden does not
vanish. It **moves to the checker.** You have replaced "trust a large solver"
with "trust a proof checker," and the checker is now the thing that must be
right. That is the CompCert move — the trusted computing base shrinks until it is
small enough to be worth arguing about — and it is a categorical improvement, not
a free lunch. Anyone who presents certificates as eliminating trust rather than
concentrating it is selling something.

So, precisely: **certificates do not close link 5. They replace it with a
smaller link.** A checked green verdict still depends on the checker being
correct, and the honest statement of the win is about the *size and
arguability* of what remains — a checker is a fraction of a solver's size, has
no heuristics, no incremental state, and no performance-driven optimisation
passes, and it can be qualified or even formally verified in a way no solver can.
That is worth a great deal and it is not the same as zero. (Real Alethe/LFSC
checkers are thousands of lines, not hundreds; and Appendix A's CompCert row is
the honest reminder that even there the TCB is the Coq kernel plus the
specification, not nothing.)

Two facts make this nearer than "Long" suggests, and both are worth a Phase-4
spike rather than a Phase-5 assumption:

- **cvc5 emits proofs** (Alethe, LFSC, and other formats, subject to build) and
  independent checkers for those formats exist. Whether the PyPI wheel's build
  supports proof production, and in which formats, is **an empirical question of
  exactly the kind this repository already answers well** — the `nl-cov`/`exp`
  finding in `SOUNDNESS.md` and the libpoly finding in `README.md` are the model.
  Verify it against the wheel; record the version, transport, and options; do not
  take this paragraph's word for it.
- The checker is small, is not stelling, and can be a separate qualified
  artifact. It does not need JAX. It does not need the transfer registry. It
  needs the query and the proof.

**The spike's real question is not whether, it is for which theories.** Proof
production is uneven across theories, and it degrades exactly where stelling
lives: `design/founding.md` targets QF_LRA at Stage 0 and **QF_NRA** from the
moment `dt` goes symbolic, with nlsat and the coverings solver doing the work —
and nonlinear real arithmetic is where solver proof support is thinnest and most
likely to contain trusted holes (steps the proof asserts rather than derives). A
certificate story that covers the linear fragment and stops is not a certificate
story for this tool. **Ask the fragment question first; it decides the whole
strategy**, and a spike that returns "yes, cvc5 emits Alethe" without naming the
theories has answered nothing.

**And note what a certificate does not touch even when it works.** A checked
proof of the wrong formula is still a checked proof. Certificates shrink link 5
and leave links 1, 3, 4, 6, and 7 exactly where they were. The trust product does
not collapse to 1 — it collapses to 6, one of which is smaller than it was. That
is worth saying before anyone budgets on the assumption that certificates end the
argument.

### 11.4 Per-framework material

Astrée and Polyspace ship framework-specific qualification kits as products
(Appendix A). That model is right and stelling should follow it eventually —
`docs/qualification/{iec61508,iso26262,do330,en50128}.md` are the skeletons. Each
answers, in its own framework's vocabulary and clause numbering, exactly one
question: **given this framework, what does stelling provide and what must you
supply?**

They are Phase 4/5 work. Writing them before the trust argument (§2.3) has
content would produce documents that map an empty evidence base onto clause
numbers, which is the genre of compliance document that makes assessors
suspicious of everything else in the file — and rightly.

---

## 12. Standards mapping

### 12.1 IEC 61508-3 §7.4.4 — off-line support tools

The base framework, and the one the SIL 3/4 audience will reach for first.

| What the standard asks | What stelling provides | Gap |
|---|---|---|
| **Tool classification** (T1/T2/T3) | **T2**, argued from architecture: stelling emits no code and touches no build artifact, so it cannot introduce a defect into the executable; it can fail to reveal one (§1.2) | **Certified parameter ranges will test this** — a tool that outputs a number the user pastes into their code contributes indirectly to the executable (§1.2's T2 exception) |
| **T2/T3: evidence the tool conforms to its specification or documentation** (§7.4.4.4) | The decision problem in one sentence plus the semantics tier (§11.1 §3); the corpus record; the witness-checking record (a positive result: each replay reproduced); the *search* record — differential, cross-solver and probe runs, with their scopes and their firings (§11.1 §7); the coverage report as a per-query scope statement. *This cell read "differential, witness-checking, and cross-solver evidence", which offered two searches as conformance evidence; a search that did not fire is not evidence that the tool conforms to its documentation (§6.1, §2.3.1)* | **Substantial today.** Everything in the middle column is Phase 2–3; at v0.1 this row is a plan |
| **Specification / manual defining behaviour and constraints of use** | `docs/qualification/constraints_of_use.md`, plus the **per-verdict coverage report** — a machine-generated constraints statement, which no other tool in Appendix A ships | Structurally none; the coverage report is Phase 2 and the content grows with the registry |
| **History of successful use in similar environments** | Retained per-release corpus records (§6.2) — the only qualification route a solo project supports unaided | **Starts empty and can only be earned.** This is why retention starts before there is anything to retain |
| **Tool validation results documented: strategy, activities, tools, results, test cases, discrepancies** | The evidence directory is exactly this, retained and versioned rather than written as a report | Format, not substance — once the directory has contents |
| **Compatibility of tools in the chain verified** | `python -m stelling` reports both transports and an external cvc5's feature set today; the stamp (§8.4) will record what a given verdict actually ran against | **Reporting is not verification.** Nothing checks that a given (jax, solver, option-set) combination is one stelling has evidence for; `TESTED_JAX_SERIES` warns and proceeds (§4.2). Open |

**The Gap column mixes two timeframes and says which.** Rows marked with a phase
describe the architecture once built; rows marked *today* or *open* describe v0.1.
The qualification package (§11.1) must carry the *today* column only — a mapping
table that quietly reads as present tense is the genre of compliance document
§11.4 warns about.

**The demand side.** IEC 61508-3's Annex A tables recommend formal methods at the
higher integrity levels — recommended (R) around SIL 2–3, highly recommended (HR)
at SIL 4, across the design and verification tables. There is currently no route
to satisfying that for a JAX program at all. **That gap is the reason this
project has a functional-safety audience and not only a research one**, and it is
the one-line argument for why the work matters to someone who has never written a
jaxpr. (Clause and table specifics: see §12.6.)

### 12.2 ISO 26262-8 §11 — confidence in the use of software tools

| Step | stelling |
|---|---|
| **Tool Impact (TI)** | TI1 when a tool malfunction cannot introduce *or fail to detect* errors in the item — available exactly when the verdict removed no work. TI2 otherwise. **The user's determination** (§1.3) |
| **Tool error Detection (TD)** | **Only the green direction ever reaches this step** — see below. Green's candidate detection mechanisms are fuzz-on-verified, cross-solver, and the cross-engine lattice (§6.1). **stelling offers no TD level for the green direction and the row below says why.** With certificates, a green verdict has a checkable artifact and the argument changes qualitatively |
| **TCL** | Falls out of TI × TD. TI1 → TCL1 regardless of TD. TI2 × TD1 → TCL1; TI2 × TD2 → TCL2; TI2 × TD3 → TCL3 |
| **Qualification methods** (if TCL2/3) | 1a *increased confidence from use* — **supported directly** via the retained corpus record (§6.2). 1c *validation of the software tool* — **inputs supplied**: the corpus, differential suite, witness-checking record, the specification. 1b *evaluation of the tool development process* and 1d *development in accordance with a safety standard* — **not supported**; stelling has no certified process and §15 says so |

**Why this row now offers no level, and what would change that.** ***It read
"a TD2 argument at best today, and the user must judge", and the correction is
not a softening — the row was arguing from the wrong quantity.*** The TD step
asks for **confidence that a malfunction or an erroneous output is prevented or
detected**. All three mechanisms named are refutation channels: each can find a
defect and none can report that there is not one, so a run of any of them that
ends quietly contributes nothing to a confidence claim. One of them,
fuzz-on-verified, is **default-off in the shipped library** and runs on no
verdict unless a caller asks for it (§2.3.1); one of them, cross-solver, is
opt-in with the solver escalation and licenses nothing when it agrees — *agreement
is not evidence* (§6.1, §11.2); and the cross-engine lattice is not built. **A
confidence level offered on that basis would be a number with nothing under it,
in the one document written for an assessor.**

What a level above the floor would need is not more mechanisms, it is **measured
detection power** — for each mechanism, the class of defect it is demonstrated to
catch, established by planting defects of that class and showing it catches them.
The tree already has that discipline and does not yet have it here: the mutation
batteries report a surviving mutant as a statement about the *suite*, the
property suite ships a per-property positive control and refuses a property whose
control cannot be demonstrated, and `fidelity.gauge` refuses to bless a gate
stack with an unexplained survivor (§2.3.1). **Detection power measured that way,
per link, is the input a TD argument is made of. Until it exists per mechanism
and per link, this document offers no TD level and the user determines one, or
does not.** Two things bound how far this row can ever go on its own: **the
determination is the user's** (§1.2, §1.3), and the clause and level numbering
here is cited from working knowledge and unverified (§12.6).

**The asymmetry lands on TI, not TD, and getting this backwards matters.** The
instinct is to say "witness checking detects malfunctions with near-certainty,
therefore TD1, therefore TCL1." That is the wrong route to the right answer, and
the wrong route is dangerous: a TD1 claim invites the assessor to scrutinise the
detection mechanism, and a *blended* TI2 × TD1 → TCL1 argument is exactly the
determination §1.2 says is not stelling's to make.

The correct route is shorter. **The red direction never reaches the TD step at
all**: used as a bug-finder, a verdict removes no work, so a stelling malfunction
cannot affect the item, so **TI1 — and TI1 is TCL1 whatever TD says.** Witness
checking does its work upstream, by making the *use* one that removes no work,
not downstream by scoring the detection.

So: TD is a question only the green direction has to answer, and §2.2's asymmetry
is why. That is the honest mapping, and it is stronger than the blended one
because it needs one determination from the user instead of two.

### 12.3 DO-178C §12.2 / DO-330

| | |
|---|---|
| **Criterion** | 3 by default; **2** if the verdict justifies eliminating or reducing another verification or development process. Never 1 |
| **The §12.2.1 exemption** | Qualification is needed when a process is eliminated, reduced, or automated *"without its output being verified."* **Witness checking verifies the output of a falsified verdict** (§2.2, both halves). The red direction is exempt (§1.3) |
| **TQL** | Criterion 3 → **TQL-5 at every software level**. Criterion 2 → **TQL-4** at levels A and B, TQL-5 at C and D. TQL-5 is the standard's lowest-rigour tier; TQL-4 is one step above it |
| **What DO-330 wants at TQL-4/5** | Tool requirements, tool operational requirements, verification of those, configuration management, and a tool qualification plan. **The tool operational requirements — what the tool does in *your* environment, for *your* use — are the user's to write.** stelling's contribution is §11.1's package, which is the input, not the artifact |

The TQL row is a genuinely useful fact for a user to know early, and it belongs
in `docs/qualification/do330.md` near the top — stated precisely, because the
tempting summary is wrong:

- **Used as an additional check** (criterion 3), stelling is **TQL-5 at every
  software level, including level A.** That is the lowest tier DO-178C defines,
  and it is the whole point: a tool that cannot inject an error into the airborne
  software is simply not where the risk is.
- **Used to eliminate or reduce other work** (criterion 2) at level A or B, it is
  **TQL-4** — not the lowest tier. That is the price of §1.3's rule, in DO-178C's
  own currency, and it is a real price.

Both facts are favourable and neither is "the worst case is the lowest tier."

### 12.4 EN 50128 §6.7

Same T1/T2/T3 classification as IEC 61508-3; stelling is T2 by the same argument.
The distinctive requirement is the ISA relationship — an independent safety
assessor reads the tool evidence directly rather than sampling it — which raises
the bar on *readability* rather than on content: every artifact should be written
for someone who has never seen the codebase. That is a writing standard, and it
belongs in §9.4 rather than as a separate compliance activity.

### 12.5 Other regimes

Sector-specific quality-system regimes generally require that software used to
develop or verify a product be validated for its intended use, and identify the
tools used with enough precision to reproduce a result. Both requests reduce to
artifacts this document already specifies: the qualification package (§11.1)
answers the first; the stamp and the reproducibility unit (§8.4, §14) answer the
second. No separate structure is needed, and none should be built speculatively.

### 12.6 A note on citations

**The standards are paywalled, and the clause numbers, table letters, and
recommendation grades in this document are cited from working knowledge of the
current editions.** The structural claims — that T2 is the class for verification
tools that cannot alter the executable; that DO-178C criterion 2 hinges on
eliminating or reducing other processes; that ISO 26262 TI1 turns on whether a
malfunction can affect the item; that §12.2.1 exempts verified output — are
load-bearing and the author is confident in them. The precise numbering,
particularly of IEC 61508-3's Annex A tables and the exact sub-clauses of
§7.4.4, is not independently verified here.

**Any citation in this document that reaches a user's qualification package must
be checked against the purchased text first.** This is the same discipline
`design/transparent-primitives.md` applies to claims about jax: a claim about an
external system carries the version it was verified against, or it is lore. A
standards citation with no verification record is lore with a clause number on
it, which is worse than none because it looks checked.

`docs/qualification/*.md` must therefore each carry the header:

```markdown
**Verified against:** IEC 61508-3:2010, clause text checked <date> by <who>.
**Unverified claims:** [list, or "none"].
```

---

## 13. What stelling is for: two questions §12 does not ask

§12 maps four frameworks that all ask one question: **what does stelling owe when
a verdict removes work?** It is the right question and it is not the only one.
Two others exist, both are asked routinely, and under both stelling is not a
supplicant seeking a category — it is **the thing already being asked for**.

The distinction from §12 is worth stating plainly, because it changes who is
persuading whom:

| | §12 asks | §13 asks |
|---|---|---|
| Question | What does stelling **owe**? | What is stelling **for**? |
| stelling's posture | A tool seeking a qualification argument | A control the framework already prescribes |
| The artifact | The qualification package (§11) | A verdict, and the constraint of use behind it |

**Scope note, and it binds this section.** Both questions are asked in every
sector that ships software into things that can hurt people. Both are *named most
precisely* in one sector's standards, and this section cites those standards
because they are the clearest instances — not because they are the target.
Third-party-software routes exist everywhere under other names (previously
developed software; proven-in-use). Simulation-credibility frameworks exist
outside that sector too. **These are sector-general question-classes, cited at
their sharpest instance.** Nothing here reclassifies the project, and §4.1's
positioning statement does not change.

### 13.1 The third-party-software question

*You are shipping software you did not develop under your own process. What do you
do about it?*

Every framework has a route. The clearest instance is **IEC 62304's SOUP** —
Software of Unknown Provenance — which is the concept most people reach for by
name, and which requires the manufacturer to specify what the item must do,
specify what it needs to run, verify its operation in the architecture, evaluate
its known anomalies against hazards, and keep doing so.

**Two things follow, and the second is the one that matters.**

**(a) stelling is not seeking a new category here. It is the prescribed control.**
The canonical way to insulate safety-involved software from a third-party
component you cannot audit by process is **static analysis of the thing itself** —
and the demand side is already documented in §12.1: IEC 61508-3's Annex A tables
recommend static analysis and formal methods at the higher integrity levels, and
there is currently **no way to do either for a JAX program.** That is a materially
stronger position than "verification tool seeking a qualification argument," and
§1.1's tool/component inversion should carry it as a third role:

| Role | stelling is… | Asked by |
|---|---|---|
| Component | — never. It ships nothing | — |
| Tool | a T2 verification tool owing evidence when a verdict removes work | §12 |
| **Control** | **the static-analysis measure the framework already asks for, applied to a language where nobody can currently apply it** | §13.1 |

**(b) The XLA precision problem has a clean home here, and it is not the anomaly
clause.** This is the useful part. There is **no anomaly to report**: JAX's
precision behaviour is documented, intended, and correct-as-specified. Filing it
against an anomaly-evaluation clause would be wrong on the facts and would look
like a smear on a dependency that has done nothing wrong.

It belongs instead against the clause that asks you to **specify what the item
must do and what it needs to run** — the functional/performance and
hardware/software requirements of the third-party item. And it lands there
perfectly, because such a requirement must be *individually testable*, and this
one is:

> *"Matmul precision shall be `F32_F32_F32` on device class X."*

That is testable, it is falsifiable at compile time (`DotAlgorithmPreset` raises
on a device that cannot honour it — §4.2), and it is exactly the shape the clause
wants. **Pinning is the deliverable, not a workaround.** A reviewer asking "what
does this component need in order to behave as you claim?" is asking a question
stelling's constraints-of-use document answers in one line, and that is a much
better conversation than "we have a known issue with GPUs."

**On clause numbers — this document does not assert them.** IEC 62304 is
paywalled and the author has not read the clause text. What is reportable: the
European Notified Bodies association's published FAQ on EN 62304 states that *"for
SOUP items, the following specific requirements apply clauses: 5.1.7, 5.3.3,
5.3.4, 5.3.6, 6.1, 7.1.2, 7.1.3, 8.1.2"*, and that *"clauses 5.3.3, 5.3.4 state
specifications requirements, clause 5.3.6 describes the need to verify SOUP
operation."* Beyond the list, three secondary sources consulted for this document
**disagree with each other** about which clause carries which title — in
particular about whether anomaly-list evaluation is 7.1.2 or 7.1.3, and one
source's claim that 5.3.4 is a "verification strategy" is contradicted by the
Notified Body FAQ, which puts SOUP verification at 5.3.6. **Three sources, three
numberings, one standard** — which is §12.6's rule arriving exactly where it was
predicted to. The argument above does not depend on the numbering and must not be
made to: cite the *list* and the *concept*, and let anyone who needs a clause
number read the standard they have paid for.

### 13.2 The simulation-credibility question

*You used a simulation as evidence for a decision. Why should anyone believe the
simulation?*

This is not tool qualification and it does not produce a TQL or a tool class. It
is a risk-informed credibility framework: **model risk sets credibility goals, and
evidence is scaled to them.** The clearest instance is **ASME V&V 40-2018** and
the FDA guidance built on it, *Assessing the Credibility of Computational Modeling
and Simulation in Medical Device Submissions*.

Unlike §13.1, this one is verifiable from a primary source, and it was:

> **Verified against:** the final guidance PDF (`fda.gov/media/154985/download`),
> read 2026-07-16. Quotes below are verbatim from it. **Unverified claims:** none —
> except ASME V&V 40-2018 itself, which is paywalled and is described here only as
> the FDA guidance describes it.

| Claim | Status |
|---|---|
| Issued as **final** guidance | *"Document issued on November 17, 2023."* Draft was December 23, 2021 |
| V&V 40 is FDA-recognised | *"The **FDA-recognized standard** American Society of Mechanical Engineers (ASME) V&V 40…"* |
| Three pillars | *"…framework for assessing verification, validation, and uncertainty quantification (VVUQ) activities…"* |
| Relationship to V&V 40 | *"This guidance uses key concepts of ASME V&V 40-2018 but provides a more general framework…"* |

**Three things are worth carrying, and one correction.**

**(a) §1.3's rule survives translation into a framework it never surveyed.** The
guidance defines **model influence** as *"the contribution of the computational
model relative to other contributing evidence in addressing the question of
interest"*, **decision consequence** as *"the significance of an adverse outcome
resulting from an incorrect decision"*, and **model risk** as *"the possibility
that the computational model and the simulation results may lead to an incorrect
decision that would lead to an adverse outcome."* Risk is influence × consequence,
and credibility evidence is scaled to it.

That is **§1.3, in a different vocabulary**: *how much does this replace, and what
happens if it is wrong.* The rule was derived here from DO-178C's §12.2.1 and ISO
26262's TI, and it reappears intact under a framework built for an entirely
different purpose by people who had never heard of this one. **A rule that
independently reappears is more likely to be the real structure than a rule that
had to be argued for**, and that is the best evidence available that §1.3 is
right rather than merely persuasive.

**(b) The framework's definition of verification is stelling's job description.**
Verbatim: code verification is *"the process of identifying errors in the
numerical algorithms of a computer code"*, and Category 1 credibility evidence is
*"testing to confirm that numerical algorithms and associated code have been
correctly implemented without errors."* Nobody involved in writing that had JAX or
SMT in mind.

Note the word **testing** — and note it honestly. The framework's expectation for
this category is testing, and a *proof* is not what it is expecting. That is an
opportunity and a friction at once: a proof is stronger evidence for exactly the
goal the category names, and it is also non-standard evidence that a reviewer has
no rubric for. This is the one place where stelling's story is *ahead* of the
framework rather than behind it, and the honest way to present it is as an
argument to be made, not a box to be ticked.

**(c) Link 7 gets a better regime here than anywhere else in this document.**
Under §12's frameworks the precision nondeterminism is a boundary: declared,
carried by the user, unclosed. Under a VVUQ framework it is a **numerical
uncertainty**, and uncertainty quantification is an entire pillar with existing
machinery for round-off and numerical error. Link 7 stops being *"declared and
carried"* and becomes **"quantified and reported"** — a thing you put a number on,
which is what the framework wanted from you anyway. That is a strictly better
home, and it is available today, with no roadmap item.

**(d) The correction.** Earlier framing of this material held that the FDA
endorsed V&V 40 for qualifying computational models through the **Medical Device
Development Tools** programme, *"which is a pathway a tool can actually walk."*
Reading the guidance: MDDT is for *"CM&S-based tools for developing or evaluating
a medical device… submitted to CDRH as a proposal to be considered for the Medical
Device Development Tools (MDDT) Program… as a **non-clinical assessment model
(NAM) for predicting device safety, effectiveness, or performance**."* **A model
is what gets qualified.** stelling is not a non-clinical assessment model and does
not predict anything about a device; it is the thing that verifies the model's
code. So MDDT is a pathway for a *user's model*, not for stelling, and this
document should not have implied otherwise.

#### The scope finding, which is better than expected

The guidance's scope section is the most interesting paragraph in it for this
project, and the **final** differs from the draft in a way that matters:

> *"This guidance document is applicable to first principles-based models (e.g.,
> physics-based or mechanistic models)… This guidance is not intended to apply to
> **standalone** statistical or data-driven models such as standalone regression,
> machine learning or artificial intelligence-based models. We recognize that there
> is no clear delineation between first principles and statistical/data-driven
> models, and that **hybrid models using both methods are possible. For hybrid
> models, this guidance is intended to apply to the first-principles model aspects
> of the hybrid model only**…"*

Three readings:

- **The crowded end of the field has no home in this framework and the underserved
  end does.** Standalone neural-network robustness — the α,β-CROWN lineage
  `design/founding.md` explicitly scopes away from — is out of scope. General
  scientific JAX is squarely in it.
- **The hybrid case is carved *in*, for exactly the part stelling verifies.**
  `founding.md` targets *"certified stability of hybrid classical/learned systems"*
  and *"MLP surrogates and GNN corrections inside classical solvers."* The guidance
  applies to *"the first-principles model aspects of the hybrid model only"* —
  which is the classical solver, which is the part stelling handles and the NN
  verifiers do not. The roadmap and the scope line up without either having been
  written for the other.
- **This is a finding, not a plan.** It says a framework exists whose stated needs
  match what stelling produces. It does not say anyone is asking for it — that is
  §6 of the work order's question and it lives in `design/value-model.md`, where a
  claim about demand can be falsified rather than admired.

---

## 14. Configuration management and reproducibility

Every framework asks the same question in its own words: *can you retrieve the
exact tool that produced this result?* For stelling the question is harder than
for a component, because the answer is not a version — it is a tuple (§8.4).

**What is already solved** (`SECURITY.md`, `README.md`):

| | |
|---|---|
| **Immutable releases** | PyPI is immutable; sdist and wheel SHA-256 per release |
| **Provenance binding** | PEP 740 attestations via Trusted Publishing bind every artifact to the exact commit and workflow run. No API tokens exist anywhere. Verifiable independently via the PyPI Integrity API |
| **Contribution provenance** | DCO sign-off per commit; no CLA |
| **Licence provenance** | REUSE-compliant; `reuse lint` in CI |
| **Citation and version identity** | `CITATION.cff` |

This is a stronger supply-chain story than most commercial tools ship, and it
answers the configuration-management half of every framework's tool requirements
with no further work. It should be said plainly in the package rather than
buried.

**What the architecture adds:**

- **The reproducibility unit is the stamp** — `(stelling, jax, solver,
  solver_version, transport, option_set, solver_features, query_hash)`, plus
  `opaque_params` for the queries the hash cannot fully distinguish (§2.6).
  Documentation must never say "pin stelling for reproducibility" (§8.4).
- **The proof cache is a configuration-management artifact.** `SOUNDNESS.md`'s
  "cache the proof, not the report" rule is a CM rule: a cache keyed on
  `query_hash` may legitimately hit across files, so a rendered verdict re-derives
  its file/line pointers from the *current* jaxpr's `source_info` and never from
  the cache. The first violation reports a line number from someone else's file —
  and in a CM context it does worse: it produces evidence that points at code
  that never ran.
- **Evidence is versioned and append-only.** `evidence/` is in git, retained
  forever, and never edited retroactively (§6.2, §7.4).
- **`python -m stelling` is already the configuration report**, and it should be
  described as one rather than treated as a smoke test. It reports which optional
  dependencies are installed, both cvc5 transports, and the feature set an
  external binary was built with — which is exactly the answer to *which cvc5*, a
  question with more than one answer at a fixed version number. What it does not
  yet do is emit that report in the machine-readable form a stamp needs; that is
  Phase 0's work, and it is a serialisation of something that already exists
  rather than a new capability.

---

## 15. Process compatibility

*§15's and §16's subsections were numbered* `14.x` *and* `15.x` *— one parent
behind, each — from the edit that inserted §14. Because §14 has no subsections
of its own, the two runs did not collide and nothing looked wrong on the page;
what it cost was six in-document cross-references pointing at headings that did
not exist — §16.1 at four sites, §16.2 and §16.3. The references were right and
the headings were wrong, so the headings are renumbered here, and every §N.M in
the file now resolves against a heading — every one that does not is a clause
of an external standard, and there are six of them: DO-178C §12.2.1,
IEC 61508-3 §7.4.4 / §7.4.4.4, ISO 26262-8 §11.4.5 / §11.4.5.2, EN 50128 §6.7.
Correctly so. Section NUMBERS are the one identifier in this document that no
anchor, link or test holds, which is why this had to be found by counting.*

*That list named DO-178C §12.2 as a seventh, and §12.2 DOES resolve — against
`### 12.2 ISO 26262-8 §11 — confidence in the use of software tools`, a heading
about a different standard that happens to carry the same number. Re-driven
over every §N.M in the file: exactly the six above fail to resolve, and the
list is now those six. The collision is the more interesting half. An external
clause reference that lands on a local heading by number is invisible to
counting in the direction counting looks — so §12.2 went onto an exception list
to explain an absence that was not there, and the exception is the only thing
that was wrong: every §N.M this file does not resolve is still an external
clause.*

### 15.1 What stelling does not have

stelling is a solo open-source project. It does not operate under a certified
development process, and this is stated plainly, not apologetically. A certified
process requires independent review, management review, and role separation that
one person structurally cannot provide — one person cannot be independent of
themselves.

Two ISO 26262-8 qualification methods are therefore **unavailable**:

- **1b, evaluation of the tool development process** — requires a documented,
  audited process to evaluate.
- **1d, development in accordance with a safety standard** — requires developing
  the tool under one.

Saying so plainly is strictly better than the alternative. A user's assessor will
determine this in ten minutes; the only question is whether they learn it from
stelling's documentation or discover it after relying on an implication.

### 15.2 What substitutes

| A certified process would give | stelling's equivalent |
|---|---|
| Document and version control | Git, tagged releases, immutable PyPI, PEP 740 attestations binding artifact to commit |
| Change control | PRs, DCO sign-off, `CHANGELOG.md`, the ledger for anything verdict-affecting |
| Verification of the tool | Replay and the corpus record (positive results), plus a retained record of the searches — differential, cross-solver, cross-engine lattice (§6.1) — each with its scope. *Listing the searches as "verification" was the substitution this whole table exists to avoid: they retain what was looked for, not what was established (§2.3.1)* |
| Defect management | GitHub issues; **plus the ledger, which does something no defect tracker does: retroactive invalidation of evidence already produced** (§7) |
| Independent review | Open source; every tier argument is public and attackable. Not equivalent to independent review, and not claimed to be |
| Requirements traceability | The specification is one sentence (§11.1 §3); the transfer registry and its documents are the detailed design; CI enforces that neither drifts (§5.5) |
| Configuration identification | The stamp (§2.4) — finer-grained than a certified process typically requires, because a version number is not enough here (§8.4) |

**Two of these are stronger than the certified-process version, and it is worth
being specific about which, because the honest accounting of the gap is more
persuasive when it is accompanied by an honest accounting of the surplus:** the
supply-chain provenance (§14) exceeds what most certified toolchains can
demonstrate, and the ledger's retroactive invalidation has no equivalent in any
process framework — those frameworks assume the tool's past outputs stay valid,
and provide no mechanism for withdrawing them when they do not.

### 15.3 The route that is available

**Increased confidence from use** (ISO 26262-8 method 1a; IEC 61508-3 §7.4.4.4's
"history of successful use") is the route a solo open-source project can support,
and it is well-precedented — it is how most of the world's compilers and tools
enter safety pipelines. It requires exactly what §6.2 specifies: a retained,
versioned, honest record, **including the malfunctions**.

The record's honesty is the load-bearing part. A usage history with no observed
defects over several years reads to an assessor as evidence that nobody looked —
because for any tool of this complexity, it is. The ledger's function in the
usage record is therefore the opposite of what it looks like: **entries are the
evidence that the record is real.**

---

## 16. Ecosystem

### 16.1 The schema layer

`stelling.evidence` is the stable, importable, **jax-free and solver-free**
surface for anyone who consumes stelling's output: the verdict schema, the
coverage schema, the ledger schema, and `stelling audit`.

The reasoning that put `stelling.ir` behind the jax boundary applies here and is
stronger: an assessor validating evidence, a CI job gating a release, or a
downstream project auditing its inherited verdicts must be able to do so without
installing a JAX stack. Evidence tooling that requires the tool's dependencies is
evidence tooling that the people who most need it will not run.

**Part of the base install, no extra, and stdlib only.** This is not a
preference — it is `pyproject.toml`'s `dependencies = []` and `README.md`'s "zero
required dependencies", which `stelling.evidence` must honour like every other
module. It is why the ledger has a generated JSON face (§7.1): a module whose
selling point is that it imports with nothing installed cannot be the module that
introduces stelling's first hard runtime dependency. Authoring tools that read the
YAML source belong in the dev group, alongside `reuse` and `pre-commit`.

### 16.2 Downstream harness libraries

The ecosystem play is not other verifiers — it is **libraries that ship their own
harnesses**. A `jax-cfd` that ships stelling harnesses for its stencil kernels,
a `diffrax` that ships them for its steppers: each is a project whose users get
verified kernels without writing a harness, and each is a corpus contributor with
its own release cadence.

Such a project inherits:

- the schema layer (`stelling.evidence`);
- the verdict semantics and everything in §4's boundary language;
- the harness authoring standard, including controls and obligations (§10.8);
- the constraints of use.

And must provide:

- **its own harnesses, and its own controls for them.** Link 1 does not transfer.
  A harness that expresses the *library author's* intent is still a harness whose
  correspondence to the *user's* intent is unestablished;
- **its own retained evidence**, at its own versions;
- **its own positioning statement.** "Verified with stelling" is not a claim a
  downstream project may make without stating which properties, at which stelling
  and solver versions, under which arithmetic, with which coverage. Unqualified
  "formally verified" badges are the exact failure mode this whole document
  exists to prevent, and they would do more damage to stelling's credibility than
  a soundness bug — a bug can be logged and fixed.

### 16.3 Citing a verdict: the two modes

The one question a downstream project or a user's assessor will actually ask is
*may I reuse this verdict, or must I re-run it?* The answer is mechanical, which
is the whole payoff of content addressing:

| Mode | Condition | Permitted |
|---|---|---|
| **Cite** | The full stamp matches — same stelling, jax, solver, version, transport, option set, feature set; **the query hash still matches a current trace**; **and `opaque_params` is empty** (§2.6) | **Cite the stored verdict.** It is the same claim about the same program under the same assumptions |
| **Re-derive** | Anything in the stamp moved, the query hash changed, or `opaque_params` is non-empty | **Re-run.** No exceptions and no tolerance parameter |

(The modes are named **Cite** and **Re-derive**, not "replay" and "re-run" —
*replay* means executing a witness everywhere else in this document (§2.2), and
reusing that word for "do not execute anything" would invert it in the one place
a reader is least able to check.)

The `opaque_params` condition is the non-obvious one and it is
`design/transparent-primitives.md`'s caveat arriving exactly where it was
predicted to: a hash match means "the same program **up to the contents of any
`OpaqueParam`**", so for a query carrying a `custom_vjp` whose `bwd` is the thing
under test, a hash match is not identity. Citing on a bare hash match is the
forbidden keying, and it would look entirely reasonable at the moment someone did
it.

There is no third mode, and deliberately no "close enough." A component library
needs a tolerance discussion here because numerical outputs can be *nearly* equal
in a meaningful way. **A verdict cannot be nearly the same verdict.** The hash
matches or it does not; the option set matches or it does not. `stelling audit`
decides it, and the absence of a judgement call is the feature.

---

## Appendix A: Landscape — what comparable tools do about qualification

The case studies that matter for stelling are not other JAX libraries. They are
the tools that have already had to answer "why should a safety assessor believe
your output?"

| Tool | Verifies | Trust story | Qualification artifacts | What stelling takes |
|---|---|---|---|---|
| **Kani** | Rust, via CBMC; harness-driven, bounded | Trust extends to rustc's backend, stated openly | None. Research/engineering tool | The harness model; `cover`; **the honesty about extending trust to the compiler backend** — stelling makes the same bet on XLA and says so (link 7) |
| **CBMC** | C/C++, bit-precise BMC | Long deployment history; traces are concrete | No kit. History-of-use arguments made by users | Counterexample concreteness as the trustworthy half |
| **Astrée** | C, sound abstract interpretation; no false negatives for runtime errors | Soundness by construction + a commercial process | **Qualification support kits sold per framework** (DO-178, IEC 61508 to SIL 4, ISO 26262 to ASIL D, EN 50128) | The induction/widening strategy (`founding.md` names "the Astrée move"); **the per-framework kit as a product shape** (§11.4) |
| **Polyspace** | C/C++/Ada, abstract interpretation | Same shape as Astrée | Certification/qualification kits sold per framework | Same |
| **SPARK / GNATprove** | Ada subset, deductive | Proof + a language designed for it | Qualification material available commercially | **The adoption levels** (stone/bronze/silver/gold/platinum) — a user-facing statement of *what you get for what you put in*. **The best existing model for what stelling's coverage report should feel like**: not a limitation list, a ladder |
| **CompCert** | Is a C compiler; its *compilation* is proven | The proof is the argument; the TCB is the Coq kernel + the spec | A qualified variant is sold commercially | **The TCB-shrinking move** — the strategic core of §11.3 |
| **Frama-C / WP; TrustInSoft Analyzer** | C, deductive + AI, ACSL specs | Open core; commercial qualified derivative | Kits sold with the commercial analyser | The contrast: **ACSL is a spec language to design, document, and qualify.** stelling's "checkify as spec" (`founding.md`, Stage 1) is a scope decision with a real but narrow dividend — **one semantics to document instead of two.** Not zero: checkify's error conditions *are* the spec, so their semantics are jax's, version-dependent, and owed the same treatment as any primitive (§5.2) — including a home in `docs/semantics/`. And a checkify change that flips a verdict is a soundness event under `SOUNDNESS.md` whoever's bug it is |
| **α,β-CROWN and the NN-robustness lineage** | Neural network robustness properties | Competition-driven credibility (VNN-COMP); some verifiers emit certificates | No safety-standard kits | Nothing directly — `founding.md` scopes stelling deliberately away from this. It is in the table to mark the boundary, and because the competition-as-credibility model is a real alternative to a usage record |
| **dReal** | Nonlinear reals, δ-decidable | δ-sat is an honest verdict about physical properties; witnesses are boxes | None | Commitment 2's stance; the future backend; **"δ-sat" as a verdict that names its own approximation** — the same instinct as the `arithmetic` field |
| **z3 / cvc5** | Are the solvers | SMT-COMP record; enormous deployment; **both unqualified** | None. Both can emit proofs in checkable formats (subject to build) | §11.2 is this row. The proof formats are §11.3's route |

**What the table says, read as a whole.** Every tool that has entered a safety
pipeline with a green claim did it one of two ways: **a commercial qualification
kit backed by a certified process** (Astrée, Polyspace, SPARK, TrustInSoft), or
**a proof that shrinks the trusted computing base** (CompCert). The first is not
available to a solo open-source project. **The second is** — and it is the same
route §11.3 identifies from a completely different direction, which is a decent
sign it is the right one.

The tools with no story (Kani, CBMC in most deployments) are used exactly the way
§1.3 describes: as bug-finders that remove no work. That is not a criticism of
them. It is the same wedge, and it is where stelling starts.

---

## Appendix B: Implementation priority, mapped to the founding stages

**This appendix adds no verification work and reorders none.** Phases 1–5 hang
off `design/founding.md`'s stages one-to-one, because a documentation
architecture that grows its own parallel schedule is how a project ends up with
two roadmaps and no verifier.

**Phase 0 is new, and it is new on purpose.** It sits ahead of the stage
`founding.md` titles *"build this first"*, and it is worth being exact about the
tension: `founding.md` Stage 0 says *"Deliberately excluded: registry, coverage
report, interval domain, control flow, IEEE-754 semantics, everything else. The
PoC is allowed to be a single file."* Phase 0 does not touch that list. It asks
for eight things that are **not verification** — types, schemas, a changelog, a
positioning page — chosen by one criterion: *is it unrecoverable if skipped?*
The PoC may still be a single file; it just must not emit an unstamped verdict
from it. Where that reading is wrong, `founding.md` wins and this appendix
changes.

*When this appendix was written the repository was* **pre-Stage-0**: `ir.py`,
`_jax_compat.py`, `_optional.py` and `__main__.py` existed; harness primitives,
the query object and the z3 encoder did not. **All three now exist and have
since before `v0.1.0` was tagged** — `stelling.harness` binds `any_array`,
`any_pytree`, `assume`, `assert_` and `nonvacuity` to real jax primitives;
`stelling.ir.ClosedJaxpr` is the query object and carries `content_hash()`; and
`stelling.smt.emit` is the encoder, reaching z3 through
`solvers.TRANSPORT_Z3_WHEEL` (`"wheel-bindings (smt2 text)"`) and cvc5 by wheel
or external binary. The phase list below is kept as the plan it was; it is not
a description of what the tree has.

### Phase 0 — now, before the first verdict exists

Everything here is cheap now and impossible later (see the executive summary).
Nothing here requires a working verifier.

1. **`Verdict` and `Environment` types with mandatory stamps** (§2.4, §10.2). The
   fields are already specified in `SOUNDNESS.md`; this makes them structural.
   **Ship no verdict before this exists** — the first unstamped verdict is a
   permanent hole in the audit.
2. **`evidence/soundness.yaml` + schema + the `affects` predicate vocabulary**
   (§7.2). Empty. The schema is the point: it forces event #1 to be written as a
   predicate.
3. **`SOUNDNESS.md` Log rendered from the ledger** (§7.1) — trivial while the log
   is empty, and it can never drift afterwards.
4. **`evidence/` directory, schemas, and the retention rule in CI** (§6.2). The
   first release must write its corpus record even if the corpus has three
   entries.
5. **`CHANGELOG.md`** with Soundness / Verdicts / Coverage sections (§8.3).
6. **`docs/qualification/positioning.md`** and the §1.3 rule in `README.md`
   (§4.1). One page. It is the answer to the first question every serious user
   asks, and it costs an afternoon.
7. **`Tier` enum and `TransferMeta`** (§10.1) — before there are transfers, so the
   first one cannot be registered without an argument.
8. **`design/_template.md`** — the `transparent-primitives.md` header,
   generalised (§5.2).

### Phase 1 — with Stage 0 (the proof of concept)

9. **Witness checking before construction** — a `FALSIFIED` `Verdict` cannot exist
   without a checked witness, both halves (§2.2, §10.2). Stage 0 already requires
   replay; this makes it an invariant rather than a step, and adds the in-R half.
   **Check on CPU as well as the accelerator** — one extra call, and it buys a
   differential against links 7 and 8 (§2.3).
10. **Vacuity in the type** — `VERIFIED` requires `vacuity is WITNESSED` (§10.2).
11. **Positive controls** — `stelling.control(of=…, expect=FALSIFIED)`, plus the
    hard error on syntactically-constant-true B (§10.8, gap 2).
12. **`obligations` on the verdict** (§10.8, gap 6). One field. It prevents the
    induction-base-case class and every future assume-guarantee variant of it.
13. **Exit-code discipline** (§10.9), including `--allow-unknown` as opt-in.
14. **Fuzz-on-verified** (§2.5) — the fuzzer exists at Stage 0 for the vacuity
    guard; pointing it at verified queries is a small delta, and it is **the only
    mechanism this architecture ever points at links 2 and 7** before Phase 5
    (§6.1). **It can only refute, so pointing it at those links does not defend
    them** (§2.3.1) — and as built it is default-off, so the item is not "the
    probe exists" (it does) but "it runs without being asked for" (it does not).
15. **`docs/semantics/backends/z3.md`** with the full emitted option set and the
    reason for each (§5.4).
16. **The verdict artifact written to `evidence/`** (§2.6).

### Phase 2 — with Stage 1 (the wedge: index safety on real code)

17. **Coverage report artifact** — schema, emission, retention (§6.3).
18. **Transfer registry with mandatory `TransferMeta`**; registration refuses a
    `sound`/`heuristic` tier whose argument does not resolve (§10.1).
19. **`docs/semantics/transfers/`** + `_template.md` + the CI bridges (§5.5).
20. **Corpus + `manifest.yaml` + per-release retention** (§6.2, §10.4). The
    primitive census is already Stage-1 work; **retaining its results is the
    delta**, and it is small.
21. **`stelling audit` v1** — INVALID / REVIEW / STALE (§7.3, §10.6). Implementable
    as soon as stamps and the ledger exist, and it is the most valuable command
    for anyone with a pipeline.
22. **`docs/qualification/constraints_of_use.md`** (§4.2).
23. **`docs/user_guide/harness_authoring.md`** — controls, obligations, the
    vacuity gaps (§10.8).
24. **`stelling.evidence` as a jax-free import surface** (§16.1).

### Phase 3 — with Stage 2 (floats, induction, first physics proofs)

25. **Differential testing** — already Stage 2. The addition: results land in the
    retained record (§6.2), and a failure classifies into the ledger rather than
    just going red.
26. **Tier surfacing in the coverage report and the verdict**; `soundness`
    derived by weakest link (§5.1).
27. **Cross-solver disagreement hunt on the corpus**, keyed on `query_hash` (§6.1) — the deliverable is the hunt and its scope, not a tally of agreements (§2.3.1).
28. **The cross-engine lattice as a test**, with "stelling contradicted itself" as
    a distinct exit code (§2.5, §10.9).
29. **`docs/semantics/arithmetic.md`** — link 6 in full, with the `arithmetic`
    field's meaning (§4.2). Stage 2's soundness-plumbing work (outward rounding,
    MPFR on CPU) is what this document has to describe honestly.
30. **Qualification package v1** (§11.1) — the trust argument (§2.3) with real
    content in the checked-today column.

### Phase 4 — Medium

31. **Proof caching** keyed on jaxpr hash — already on the roadmap; the CM rules
    (§14) come with it.
32. **Certificate spike** (§11.3): does the cvc5 wheel emit proofs, in what
    formats, checkable by what? An empirical question, answered the way this
    repository answers them — with a version, a transport, an option set, and a
    date.
33. **`@stability` machinery** (§10.5).
34. **Per-framework qualification documents** (§11.4) — once §2.3 has content.

### Phase 5 — Long

35. **Certificates shipped**; the checker as a separate, small, qualifiable
    artifact (§11.3).
36. **Translation validation to HLO** — link 7.
37. **Sphinx site** (§3.2).
38. **Per-transfer search reports generated from the differential suite** — what was drawn, what was not, what fell outside the bounds. *Not "verification reports": the suite refutes a tier and cannot establish one (§2.3.1).*

---

## Appendix C: Relationship to existing documentation

| Document | Status | Relationship |
|---|---|---|
| `SOUNDNESS.md` | **Excellent. Normative. Outranks this document.** | This document is the architecture that makes its promises mechanical. Two additions: the Log renders from the ledger (§7.1); the "retroactively invalid" promise becomes an evaluable predicate (§7.2). The Policy prose stays hand-written and unchanged. **One caution:** "cache the proof, not the report" is about *source locations* and makes no claim that a hash match implies semantic identity — §2.6 is where that gap is recorded rather than assumed away |
| `design/founding.md` | **Excellent. The commitments and roadmap.** | §2.3's eight links reframe *three* of the ten "Long" bullets as a trust-debt schedule and leave the rest alone. **Additions offered, none of them reorderings:** the second half of witness checking — confirm the witness is in R, not only that the assertion fails (§2.2), which ⊤'s over-approximation makes load-bearing; witness checking as a construction invariant rather than the encoder's unit test (§1.3, §10.2); fuzz-on-verified rather than fuzz-on-unknown (§2.5); the trivial-property vacuity gap (§10.8 gap 2); the induction base case as an obligation (§10.8 gap 6); and Phase 0's eight non-verification items (Appendix B). Each is small; each is cheapest now |
| `design/transparent-primitives.md` | **Excellent. The template for the whole semantics tier.** | §5.2 generalises its header — status, normativity, evidence with version and date, forcing function — into `design/_template.md` |
| `design/value-model.md` | **New in v0.2. The other half of the decision.** | This document is a cost model: it says what legitimacy costs and nothing about what a verdict is worth. The value model states the wedge's claim, the counterfactual (point methods vs. a region method), the experiment (N=5, named), and a **pre-registered falsifier** (zero real out-of-bounds ⇒ stop or re-aim). It is short and must stay short — if it becomes a second cost model it has failed. Its epistemics are §10.8's positive-control discipline turned inward: **a value claim that cannot fail is the same defect as a verdict that says more than it earned, one level up** |
| `README.md` | Good | Add the §1.3 rule and the three repeated sentences (§4.4). The cvc5 wheel/binary findings should move to `docs/semantics/backends/cvc5.md` and be maintained there, with the README linking |
| `CONTRIBUTING.md` | Good | Grows the §9 checklists as the transfer registry appears. The three ground rules stay as they are |
| `SECURITY.md` | Good | Already answers configuration management (§14). Add the input-trust boundary sentence (§4.1): stelling executes user code by construction and is not a security boundary |
| `CITATION.cff` | Good | Dual-purpose: citation and configuration identification (§14) |
| `pyproject.toml` | Good | Add a `docs` extra when Phase 5 arrives, and nothing sooner |
| `src/stelling/ir.py` | **Excellent** | The content hash is the spine of the evidence architecture (§2.6). Its docstring's account of why source locations are excluded is the same argument §7.3's STALE detection depends on. Its `OpaqueParam` docstring is the argument §2.6's caveat depends on |
| `src/stelling/_optional.py` | Good | `TESTED_JAX_SERIES` is the model for a version claim with a forcing function (§5.2). The same pattern should apply to solver claims |
| `src/stelling/__main__.py` | Good, and under-credited | It is already the configuration report §14 wants — both cvc5 transports, the external binary's built-with feature set. Phase 0 serialises what it prints; it does not add a capability |
| `tests/test_ir.py` | Good, and mis-cited if read as transcription evidence | Its round-trip is IR → dict → IR and never touches jax, so it pins *serialization*. Transcription fidelity (link 2) needs per-primitive traced comparisons — Phase 2, §6.1 |

**The pattern worth naming:** this repository already documents better than most
mature projects, and it does it with one habit — **a claim about an external
system carries the version it was verified against and the trigger for
re-checking it.** `TESTED_JAX_SERIES`, the jax-0.10.2 evidence table, the cvc5
1.3.4 option findings. Everything in §5 and §12.6 is that habit, applied to two
more external systems (solvers and standards) and made structural.

---

## Appendix D: The qualification roadmap

Three tracks, running independently. The tool does not wait for the second, and
the second does not wait for the third.

### Track 1 — the tool

Build the verifier per `design/founding.md`. No regulatory timeline touches this.
The only obligations Track 1 owes the others are the Phase 0 items — stamps,
ledger, retention — because they are unrecoverable if skipped, and each of them
is a day of work.

### Track 2 — evidence

Accumulate what cannot be bought or backfilled:

| | |
|---|---|
| **Usage record** | Retained per-release corpus results. Starts at v0.1.0 or never (§6.2) |
| **Field record** | Real projects, real harnesses, real findings. `founding.md`'s field test is this track's first entry |
| **Defect record** | The ledger. Its entries are evidence the record is real (§15.3) |
| **Independent record** | Publications, external users, reproductions by people who are not the author. The only genuinely independent evidence a solo project can accumulate, and the slowest |

Track 2 is measured in years and is the binding constraint on Track 3. It is also
almost free if the retention discipline exists from Phase 0, and impossible to
start late.

### Track 3 — user qualification

Begins when a user needs a verdict to be load-bearing. **stelling's role is to
supply artifacts, never to make the argument.** What each track owes:

| | stelling | The user |
|---|---|---|
| Tool classification | States the shape (T2; criterion 2 or 3; TI1 or TI2) | **Determines it** |
| Tool operational requirements | — | **Writes them** — what the tool does in *their* environment |
| Constraints of use | Provides, static and per-verdict | Reads, and constrains use accordingly |
| The trust argument | Provides §2.3 with the current state of each link | **Judges whether it suffices** |
| Usage history | Provides the retained record | Assesses relevance to *their* environment and application |
| Validation for their use | Provides the corpus, differential suite, replay record | **Performs it** for their configurations |
| The qualification argument | — | **Owns it** |
| The assessor relationship | None | **Theirs** |

The last row is the one that determines the shape of everything above it: **there
is no relationship between stelling and any assessor, ever.** stelling produces
artifacts; users produce arguments; assessors judge arguments. A tool that starts
negotiating with assessors on its users' behalf has taken on an obligation it
cannot discharge and a liability it cannot carry.

---

## Appendix E: Repository setup checklist

Each item is a yes/no verifiable by inspection. Organised by Appendix B's phases.

### Phase 0 — before the first verdict

**Verdict types**

- [ ] `Verdict` and `Environment` exist with **all** stamp fields from §2.4 required — including `coverage_ref` and `opaque_params` (§10.2)
- [ ] `Verdict` cannot be constructed without an `Environment` — no default, no `None`
- [ ] `Environment.solver_options` holds the **full emitted set**, and there is no code path that invokes a solver on defaults (`SOUNDNESS.md`)
- [ ] `Environment.solver_features` has **no default**, and a binary-transport `Environment` cannot be constructed without it — `SOUNDNESS.md` requires the `--show-config` set for an external cvc5 (§10.2)
- [ ] `Verdict.soundness` is derived by weakest link over `tiers` (§5.1)
- [ ] `arithmetic`, `trust_boundary`, `vacuity`, and `obligations` are fields, not documentation (§2.4)
- [ ] `Result` distinguishes VERIFIED / FALSIFIED / VACUOUS / UNKNOWN — four values, not a boolean plus a note
- [ ] `python -m stelling`'s configuration report is available as data, not only as printed text — Phase 0 serialises what `__main__.py` already discovers (§14)

**The ledger**

- [ ] `evidence/soundness.yaml` exists with `schema_version` and an empty `events` list
- [ ] `evidence/soundness.json` is **generated** from the YAML and **committed**; CI fails if it is stale (§7.1)
- [ ] `evidence/schema/soundness.schema.json` exists and defines the `affects` predicate vocabulary
- [ ] Every `affects` key corresponds to a field on the verdict stamp (§7.2)
- [ ] `SOUNDNESS.md`'s Log section is **generated** from the ledger; CI fails if they diverge (§7.1)
- [ ] `SOUNDNESS.md`'s Policy section is unchanged and hand-written
- [ ] CI validates the ledger schema on every push
- [ ] CI refuses deletion or mutation of any existing event except `fixed_in` / `resolution` — the only two mutable fields (§7.2, §7.4)
- [ ] **Nothing in `src/stelling/` imports YAML.** `pyproject.toml` still says `dependencies = []` (§7.1, §16.1)

**Evidence retention**

- [ ] `evidence/` exists, is in git, and its contents are append-only by CI rule (§6.2)
- [ ] `evidence/schema/verdict.schema.json` and `coverage.schema.json` exist
- [ ] The release workflow writes `evidence/corpus/<version>.json`, **even if the corpus is nearly empty**
- [ ] A documented rule states that evidence files are never edited retroactively

**Repository root**

- [ ] `CHANGELOG.md` exists with Soundness, Verdicts, and Coverage sections (§8.3)
- [ ] `README.md` carries the §1.3 rule verbatim and the three sentences from §4.4
- [ ] `docs/qualification/positioning.md` exists per §4.1, including the input-trust boundary

**Metadata**

- [ ] `Tier` enum with exactly `EXACT`, `SOUND`, `HEURISTIC` — three, per commitment 5 (§10.1)
- [ ] `TransferMeta` exists with `tier`, `argument_ref`, `jax_semantics_verified`, `verified_on` all required
- [ ] `design/_template.md` exists with the status / evidence / forcing-function header (§5.2)

### Phase 1 — with Stage 0

- [ ] A `FALSIFIED` `Verdict` cannot be constructed without a **checked** witness — **both halves**: the assumes evaluated concretely on it (it is in R, not merely in the ⊤-widened R′), and the assertion failing under `jit` (§2.2, §10.2)
- [ ] A `VERIFIED` `Verdict` cannot be constructed with `vacuity != WITNESSED` (§10.2)
- [ ] `stelling.control(of=…, expect=FALSIFIED)` exists and runs in the same invocation (§10.8)
- [ ] A query whose B is syntactically constant-true is a hard error, not a `VERIFIED` (§10.8)
- [ ] `obligations` exists as a field and renders next to every verdict that has one (§10.8). The induction API that populates it is Phase 3 — the field lands now so the API cannot ship without somewhere to put the base case
- [ ] Exit codes per §10.9; `unknown` does **not** exit 0; `--allow-unknown` is opt-in
- [ ] The fuzzer runs on `verified` queries **by default**, not only on
      unknown/timeout and not only when a caller passes a keyword (§2.5, §2.3.1).
      *Built and reachable as `check(..., falsify="sample")`; unticked because the
      box is about the default, and because a probe nothing runs defends nothing
      and detects nothing*
- [ ] Engine disagreement produces exit code 5, **names the candidate links, and does not assert that stelling is at fault** (§2.5, §10.9)
- [ ] `docs/semantics/backends/z3.md` lists every emitted option with its justification (§5.4)
- [ ] Verdict artifacts are written to `evidence/` per the schema (§2.6)

### Phase 2 — with Stage 1

- [ ] Coverage artifact emitted with every verdict, per §6.3's fields, retained
- [ ] A verdict rendering that omits coverage does not exist (§10.3)
- [ ] Transfer registry requires `TransferMeta`; no transfer registers without a tier
- [ ] Registration of a `sound` or `heuristic` transfer whose `argument_ref` does not resolve **fails** (§10.1)
- [ ] `docs/semantics/transfers/_template.md` exists per §5.3, with the Tier argument section mandatory
- [ ] `scripts/check_tiers.py` — doc header vs. `TransferMeta`; code wins (§5.5)
- [ ] `scripts/check_encoding_map.py` — every qualified name resolves (§5.5)
- [ ] `scripts/check_citations.py` — `[@Key]` resolves against `docs/bibliography.bib` (§5.5)
- [ ] `scripts/check_transfer_docs.py` — **a registered transfer with no document fails CI** (§5.5)
- [ ] `corpus/manifest.yaml` with source, commit, licence, exercises, rationale per entry (§10.4)
- [ ] Corpus entries default to `expected: None` — the corpus records, it does not assert (§10.4)
- [ ] `stelling audit` with INVALID / REVIEW / STALE and exit codes per §10.6
- [ ] `stelling audit` REVIEW (undecidable predicate) exits nonzero, not warn
- [ ] `docs/qualification/constraints_of_use.md` per §4.2
- [ ] `docs/user_guide/harness_authoring.md` covers controls, obligations, both vacuity modes
- [ ] `from stelling.evidence import audit` succeeds **without jax and without a solver installed** (§16.1)

### Phase 3 — with Stage 2

- [ ] Differential results land in `evidence/corpus/<version>.json`, not only in CI logs
- [ ] A differential failure classifies into the ledger (§7.2), not only into a red build
- [ ] Every transfer's tier appears in the coverage report and in `Verdict.tiers`
- [ ] A heuristic-chain `VERIFIED` never renders as a bare `VERIFIED` (§2.1 rule 3)
- [ ] Cross-solver disagreement hunt runs on the corpus, keyed on `query_hash`, and a disagreement fails loudly rather than being resolved by picking (§6.1)
- [ ] The cross-engine lattice (§2.5) is enforced by test, **including the ⊤ row**: an SMT chain containing a ⊤ may not report `falsified` on an unchecked model
- [ ] A lattice violation is triaged before it becomes a ledger event — links 6 and 7 are findings about the program, not stelling bugs (§2.5)
- [ ] The induction harness API discharges the base case **or** records it in `obligations` — Stage 2 is where induction ships, so this is where the Phase-1 field gets used (§10.8 gap 6)
- [ ] `docs/semantics/arithmetic.md` states what `real-with-margin` claims and what it does not
- [ ] Outward rounding on the checker path is implemented and **named in every sound transfer's tier argument** (§5.3)
- [ ] `evidence/releases/<version>/package.md` exists per §11.1
- [ ] The package's trust-argument section states the current state of all **eight** links honestly, **including the empty cells and link 8, which no roadmap item closes** — §2.3's table is the architecture's plan, not the release's evidence

### Phase 4 — Medium

- [ ] Proof cache keyed on `query_hash`; **file/line re-derived from current `source_info`, never from cache** (`SOUNDNESS.md`, §14)
- [ ] The cache is **not** keyed on `query_hash` alone for any query with a non-empty `opaque_params` — `design/transparent-primitives.md`'s Stage-2 caveat, enforced (§2.6)
- [ ] Certificate spike documented: does the cvc5 wheel emit proofs, **for which theories** (QF_NRA is the one that decides the strategy), in what formats, checkable by what — with version, transport, options, and date (§11.3)
- [ ] `@stability` machinery and a generated report (§10.5)
- [ ] `docs/qualification/{iec61508,iso26262,do330,en50128}.md`, each with the §12.6 verification header

### Phase 5 — Long

- [ ] Certificates emitted and checked; the checker is a separate, small artifact with no JAX dependency (§11.3)
- [ ] Translation validation to HLO; `trust_boundary: hlo` becomes reachable (§2.3 link 7)
- [ ] Sphinx site builds; API reference generated
- [ ] Per-transfer **search** reports generated from the differential suite, each carrying the scope it drew from (§2.3.1)

---

## Appendix F: Ecosystem bootstrap checklist

For a downstream project shipping stelling harnesses for its own kernels (§16.2).

### Inherited (verify it works)

- [ ] `from stelling.evidence import Verdict, audit` succeeds without jax installed
- [ ] `stelling audit --help` runs
- [ ] The project pins stelling with a lower bound and an upper bound on the major

### Provided fresh (none of this transfers)

- [ ] **Its own harnesses**, expressing the library author's intent, with the correspondence to *user* intent explicitly unclaimed (link 1 does not transfer)
- [ ] **At least one control per harness** intended to license any removal of work (§10.8)
- [ ] **Its own retained evidence** at its own versions — verdict artifacts and coverage, not just a badge
- [ ] **Its own positioning statement.** "Verified with stelling" is not a permitted claim. What is permitted: which properties, at which stelling and solver versions, under which arithmetic, with which coverage, with the verdict artifacts published
- [ ] A documented process for reviewing stelling releases against `stelling audit` before moving the pin (§8.4)

### Never

- [ ] No "formally verified" badge without the qualifiers. **This is the failure mode the entire document exists to prevent**, and a downstream project doing it damages stelling's credibility in a way a soundness bug does not — a bug can be logged, scoped, and fixed. A badge cannot be recalled from the minds of the people who read it.
