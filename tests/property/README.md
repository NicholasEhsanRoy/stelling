<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# The property suite

Generated harnesses, run through `stelling.preconditions.check`, with the
answers checked against each other and against an exact oracle. Driven by
[Hypothesis](https://hypothesis.readthedocs.io/), which is a **dev-group**
dependency and nothing else: nothing under `src/stelling/` imports it, and a
consumer installing `stelling` does not acquire it.

Read this before adding a property. It is written so that you do not have to
read the brief that produced the suite.

---

## Getting an environment

One command, and it **refuses** rather than clobber anything it did not make:

```
tools/property_venv.sh                # jax 0.11.0 -> ~/.cache/stelling-property/jax-0.11.0
tools/property_venv.sh 0.10.2         # jax 0.10.2 -> ~/.cache/stelling-property/jax-0.10.2
tools/property_venv.sh 0.11.0 /some/where
```

The default target is outside the checkout (a venv inside the tree would be a
new undecided root entry, which `tests/test_sdist_contents.py` fails on). Below,
`$VENV` is whatever it printed.

**What it refuses, and why it is not a denylist.** Three checks, cheapest
first: the two named shared jax venvs by resolved path; anything *inside* one
of them; and — the general one — any existing directory that looks like a venv
and does not carry the script's own `.stelling-property-venv` marker, plus any
existing non-empty directory that is not a venv. Re-running the script on a
target it made before is fine; that is what the marker is for.

A two-name denylist was the shape this started as, and it was measured to let
through `/home/nick/venvs/stelling-jax/subdir`, two other agents' venvs, and
**its own default target** — which already existed and which `uv venv` would
have recreated in silence. A denylist protects the venvs somebody thought to
name. The thing that needs protecting is "a venv this script did not create",
and only one of those is knowable by name.

It reads the `hypothesis` requirement out of `pyproject.toml`'s dev group, so
the version lives in one place. `stelling` itself is **not** installed into the
venv — the suite is driven with `PYTHONPATH=<tree>/src`, which is what lets one
venv be pointed at any worktree.

## Running

```
# against this tree
JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src $VENV/bin/python -m pytest -ra tests/property

# against SOME OTHER worktree or revision — the properties come from HERE,
# the code under test comes from THERE
python tools/property_check.py --tree /path/to/someone-elses/worktree
python tools/property_check.py --rev  fb34e0d

# demonstrate every positive control (each must FAIL where it is supposed to)
python tools/property_check.py --controls \
    --other-python ~/.cache/stelling-property/jax-0.10.2/bin/python

# the cross-series differential needs two interpreters
STELLING_PROPERTY_OTHER_PYTHON=~/.cache/stelling-property/jax-0.10.2/bin/python \
  PYTHONPATH=$PWD/src $VENV/bin/python -m pytest -ra \
  tests/property/test_cross_series.py
```

**The second interpreter needs `hypothesis` too, not just the other jax
series**, so build it with `tools/property_venv.sh 0.10.2` and do not reach for
a shared jax venv. The child runs `_corpus.py` as a script, `_corpus` imports
`_grammar`, and `_grammar` imports `hypothesis` at module scope — the corpus
itself is seeded by `random.Random`, but the module it lives beside is not.
Measured, pointed at a jax 0.10.2 venv without hypothesis: the child dies with
`ModuleNotFoundError: No module named 'hypothesis'`, the property's three
anti-vacuity guards never run, and the control is reported as a *wrong failure*
rather than as a demonstration. This is the canonical copy of that command —
`ci.yml`'s own comment names this file as where it lives — and it said
`/path/to/venv-with-the-other-jax-series` for as long as the instruction was
wrong.

Budgets are chosen by `STELLING_PROPERTY_PROFILE`:

| profile | seeds | budget | database | what it is for |
|---|---|---|---|---|
| `ci` (default) | derandomized | 1x | none — forbidden alongside `derandomize` | the per-push job |
| `dev` | random | 4x | `.hypothesis/` | working on a property |
| `nightly` | random | 40x | `STELLING_PROPERTY_DB` if set | a scheduled sweep |

`STELLING_PROPERTY_SCALE` multiplies any of them.

### The `ci` profile is a rot detector, not a defect finder

Say it that way round, because the per-push job's log lists what it *runs* and
a reader will otherwise take that for a list of what it *rules out*.

**A firing positive control is not evidence that the budget is adequate.**
Measured on this tree: the `widen` control fires at `ci`. The same property,
against a *blatanter* mutant — a wide declared box collapsed to its midpoint in
`interval.from_bounds` — did not, at 250 examples:

| under the midpoint mutant | verdict |
|---|---|
| `x ∈ [-0.25, 0.25] ⊢ x >= 0.0` | UNKNOWN |
| `x ∈ [-2.25, 2.25] ⊢ x >= 0.0` | **VERIFIED** — widening *proved* it |

A three-line probe hits that instantly. So `conjunct` and `widen` now draw
**1000** rather than 250: both mutants are caught at 1000 and neither was at
250, and the cost is **+21 s** per push (4.2 s → 14.7 s and 4.2 s → 15.3 s).

That moved the boundary. It did not change the instrument. The reordering
property's cross-kind clause is falsifiable — supply the shape as an `@example`
under an assume-pins-to-`1e9` mutant and it fires — and the search does not
reach it at `ci`, at ×4, or at ×16. The whole suite at ×4 costs **172 s**
against ~60 s and is still green against a mutant a three-line probe catches.

What the per-push job does buy, and it is worth having: the strategies still
draw their boundary classes, the registered mutations still apply, every
property still has a control, nine controls still fire, and the suite still
runs at all. **That is rot, caught on the push that causes it — not a soundness
argument.**

### Landing this suite withdrew a repo-wide pin

`tests/test_skip_inventory.py::test_no_session_skip_is_undisclosed` claims the
whole suite's skip set is complete. It **withdraws that claim** — by skipping,
with its own reason — on any session that reported an `xfail`, and this suite
ships one:

```
the completeness pin is WITHDRAWN, not passed: this session reported
1 test(s) as xfailed.
```

Measured on the whole tree with hypothesis installed: `2470 passed, 13 skipped,
1 xfailed`, exit 0, that pin among the skips. It is the pin's own rule
(disclosed ⇒ withdrawn, never failed — the same cut `N deselected` gets), and
it is not a defect in the pin. It is a consequence of this suite that nobody
wrote down, and it is written down here, in `CONTRIBUTING.md` and in the
`property` job's own comment block.

**CI is unaffected**, measured rather than assumed: none of the three whole-suite
lanes installs hypothesis, so every module under `tests/property/` gates at
collection, no xfail is reported, and their `verdict=made` assertion holds. The
pin is off for exactly the sessions that *can* run the property suite. It comes
back the day the wrap remedy lands and the marker in `test_oracle.py` is
deleted — narrowing the session does not bring it back, by the same rule.

**There is no nightly job.** Adding one needs a `schedule:` trigger in
`.github/workflows/ci.yml`, and that file was being edited concurrently when
this suite landed, so the change was kept to a single appended job. Until a
trigger exists, the nightly recipe is a command somebody has to type:

```
STELLING_PROPERTY_PROFILE=nightly STELLING_PROPERTY_DB=/some/cached/dir \
  PYTHONPATH=$PWD/src $VENV/bin/python -m pytest -ra tests/property
```

---

## What a metamorphic property is here

A metamorphic property needs **no ground truth**. It takes a harness, changes
it in a way whose effect on the *answer* is known, and checks that the tool's
two answers stand in the relation the change implies.

> Conjoining `x >= lo`, where `lo` is `x`'s own declared lower bound, cannot
> add information — so the verdict must not become *easier to prove*.

That is the whole shape. No oracle, no reference implementation, two runs.

It matters here for a specific reason. The defect this suite was built around
lives in the **translation** of the user's program to the jaxpr: an
out-of-dtype-range integer literal is reduced mod `2**bits` before any stelling
primitive binds it. An oracle that *executes* the user's program executes the
same broken translation and cheerfully agrees with the wrong answer. Two runs
that should agree still disagree. **Where a defect is in the translation,
metamorphic properties are strictly stronger than an execution oracle.**

### The direction vocabulary, which every property uses

* **toward VERIFIED** — the changed run gained proving power it should not
  have. The catastrophic direction: a wrong VERIFIED is a claim a user acts on.
* **toward REFUTED** — the changed run gained refuting power it should not
  have. Bad, recoverable.
* **neither** — a precision difference in the safe direction; the tool withheld
  more. A power gap, **not a defect, and never asserted against.** A property
  that forbade this would report the tool being careful.

Most properties as first posed are too strong, and the interesting work is
finding the version that holds. Three in this suite were rewritten, one was
refused outright, and one was refused and then restored when the mutation its
refusal said did not exist turned out to; each docstring records the refusal
and its counterexample next to the clause that replaced it. See "Refusals"
below.

## Why the oracle property is one-sided

`test_oracle.py` enumerates the declared box and evaluates the obligation *as
the user wrote it*, in unbounded Python integers.

* Finding one admitted point where the obligation is false **refutes** a
  VERIFIED. The verdict is wrong, full stop.
* Finding none **confirms nothing**. A VERIFIED is a claim about every point of
  a set; this looked at the points it enumerated, on the harnesses it happened
  to generate.

So a green run means *"no counterexample was found in what was searched"* and
never *"the tool is right"*. Everything the file asserts is of the first kind,
and the same asymmetry runs the other way for
`test_a_refuted_is_false_at_some_admitted_point`: one admitted point where the
predicate is true refutes a `violated-over-set`; none confirms it.

Two consequences worth keeping in mind:

* **integers only.** `SOUNDNESS.md` records that real mode judges floats in
  exact real arithmetic while integers are judged execution-faithfully. A float
  harness can be correctly VERIFIED in ℝ and violated by IEEE execution — the
  declared posture, not a defect. An oracle pointed at floats measures the
  documentation.
* **the box must be enumerable.** `_grammar.declared_points` returns `None`
  rather than a partial answer when the product of the declared boxes exceeds
  4096, and the caller discards the example. A refusal to answer, never a
  silent "no".

---

## Adding a property

1. **Write the version that holds, not the version you were handed.** Ask what
   the tool is *entitled* to do under your change. Losing precision is always
   allowed. If your property forbids it, it will report the tool being careful
   and somebody will delete your property rather than fix a defect.
2. Put it in `test_metamorphic.py` (two runs related to each other) or
   `test_oracle.py` (one run against ground truth). Draw from `_grammar`:
   `integer_specs` where you need the exact oracle, `general_specs` where you
   need floats, size-0 shapes, casts and connectives and do not need one.
3. **Draw everything from one strategy, as a tuple.** `st.data()` is convenient
   and it makes `@example` impossible, which you will want (step 6).
4. **Assert a census floor inside the test**, after the search:

   ```
   census = _runner.Census("metamorphic/my-property")
   ... census.draw() / census.skip(...) / census.tag("compared") ...
   search()
   census.require(compared=40, my_thing_compared=40)
   ```

   Not in a comment, not in a separate test that could be deselected or
   reordered away — in the body, so the test cannot pass without having looked
   at something. The floors are tripwires for a search that collapsed, not
   claims of thoroughness; set them well below what you measure and write the
   measured number in the docstring.
5. **Say what it does not cover**, in the docstring, in as many words. The
   module docstrings have a `NOT covered:` block; extend it.
6. **Give it a positive control** — see below. A property whose control cannot
   be demonstrated does not ship. That is a rule, not a preference; one
   property is out of this suite under it. **And look for the mutation before
   you write that there isn't one** — a second property was dropped on the
   unrun claim that no one-line mutation makes it fail, and the mutation turned
   out to exist. The rule is "the control could not be demonstrated", never
   "the control was not looked for".

### How to give it a positive control

A positive control is a **place where your property is known to FAIL**, plus a
way to run it there. It is what separates *"this property found nothing"* from
*"this property's strategy generated nothing"*, which otherwise print the same
green line.

Add an entry to `positive_controls.py`:

```
Control(
    name="my-property",
    nodeid="tests/property/test_metamorphic.py::test_my_property",
    kind="commit",            # or "mutant"
    at="fb34e0d",             # the last commit that still had the defect
    why="one sentence naming the defect, and the commit that fixed it",
    expect_message="toward-VERIFIED",   # a substring the failure must carry
)
```

then demonstrate it:

```
python tools/property_check.py --control my-property -v
```

The tool materialises `<at>`'s `src/` with `git archive` (not `git worktree` —
this repository is worked on by several agents at once), runs **your current
property** against it, and asserts the run comes back RED with that message.
`CONTROL DID NOT FIRE` exits non-zero.

`--rev` and `--controls` therefore need a **git checkout**. From an unpacked
sdist there is no history to materialise and only `--tree` works.

**`kind` is load-bearing.** A `commit` control is a defect somebody *shipped* —
evidence that the class is real. A `mutant` control is a defect somebody
*invented* — evidence only that the property can see something. Both are
useful; conflating them is not. A mutant carries a `Mutation(path, old, new)`
applied to a scratch copy, and `test_suite_disclosure.py` asserts statically
that `old` still occurs exactly once, so a mutant that stops matching fails on
the commit that moves the line rather than the next time somebody runs the
controls.

**A control that fires does not say *which part* of your oracle it
demonstrated.** `tools/property_check.py` checks two things — the run came back
RED, and the failure carried `expect_message` — and neither of them knows that
your oracle is a conjunction. If it has three clauses and the tree you point at
violates one, the control is green and two clauses have been demonstrated by
nothing.

That is not hypothetical here. `test_cvc5_protocol.py`'s oracle is
`exit 0 AND nothing truncated AND the model equals the value records read`, and
`0ad22bb` — the commit `cvc5-flat` and `cvc5-stateful` both point at — violates
the middle clause and only the middle clause: measured over the flat leg's own
1500 `ci` examples with all three evaluated independently, **5 violations, all
of them clause (2)**. The other two are demonstrated by two controls added for
the purpose — `cvc5-exit-tell` at commit `8ef8f75`, and `cvc5-phantom-model`, a
mutant — whose `expect_message` is the clause's own sentence rather than the
leg tag `[flat]`, because `[flat]` is stamped on all three messages and would
have been satisfied by the failure `cvc5-flat` already finds.

So: if your oracle is a conjunction, say in the control's `why` which conjunct
its tree exercises, and register a control for each of the rest or write down
that you did not. The measurements for this one are in `_judge`'s docstring in
`test_cvc5_protocol.py`.

**Reach for `git log -S` before you reach for a mutation.** Both of those
started life as mutants, on the ground that the defect had never been in this
tree, and for clause (1) that was false: `git log -S "or proc.returncode != 0"
-- src/stelling/solvers.py` names the commit that ADDED the guard, so its
parent carries the defect and the entry is a `commit` control. For clause (3)
the same question has an empty answer — `git log -S "sorted(set(values))"`
finds nothing — and that is what makes `cvc5-phantom-model` honestly a mutant.
`kind` is only load-bearing if the question is actually asked.

**If the unbiased search cannot build the shape, pin it and say so.** Several
controls here needed an `@example`, because the shape is a conjunction of
conditions the wide grammar does not stumble into (a size-0 declaration beside
a rank-0 one, under a load-bearing assume, all at once — 2500 examples did not
produce one). Pinning it is legitimate; hiding the fact is not. Each pinned
example's docstring says the search did not find it.

---

## What this system cannot reach

Stated with numbers, from this project's own catalogue of **153** defects, so
that a green run cannot be over-read.

| | count |
|---|---|
| would have been caught by a generated-harness suite | **61** |
| would **NOT** | **81** |
| unclear | 2 |

And the cross-tab that actually decides it:

| trigger | CODE-VERDICT | CLAIM |
|---|---|---|
| (a) generated harnesses run through the verifier | **50** | **4** |
| (b) only across two configurations/versions | 1 | 2 |
| (c) only by reading prose/docs/figures | 0 | 6 |
| (d) only by a human comparing an artifact to its source | 0 | **27** |
| (e) something else (source mutation, hand-built IR, fault injection) | 29 | 19 |
| **total** | **80** | **58** |

**Generated input is the trigger for 50 of 80 code-verdict defects and for only
4 of 58 claim defects.** More than a third of this project's catalogued defects
are CLAIM defects found by a human comparing an artifact to its source — a
docstring against its implementation, a figure against the corpus it was
computed over, a count against the registry it describes, a falsifier against
the claim it is supposed to falsify. **Nothing in this suite reads a docstring.**

Two further limits worth naming:

* the ~29 code-verdict defects in bucket (e) need **mutation testing and
  hand-built IR**. A harness generator plus a mutation rig covers 79 of 80
  code-verdict defects — but that second rig is not Hypothesis, and 79-of-80
  must not be quoted as a figure about this suite.
* bucket (b) = 5 is a **floor, not a measurement**. Three of the five were
  found in one pass by installing the other jax series. The cross-series
  property here is a first instrument, over a 225-harness corpus; it is
  under-explored, not nearly exhausted.

**This does not replace the audit discipline.** It is a cheap, always-on net
under one class of defect. The class it cannot see is the larger one.

---

## Refusals, and what they cost

One property is **not here**, and the reason is the useful part. A second was
 dropped and has been restored; that story is below it, because it is the more
 instructive of the two.

**`refine=None` and `refine="affine"` must not disagree on a definite verdict.**
Sound. Also *incapable of failing on this tree*, which makes it exactly the
kind of green line this suite exists to prevent. The affine refinement only
ever visits obligations the interval leg left UNDECIDED, so an obligation the
interval leg called `discharged` or `violated-over-set` keeps that answer
whatever the refinement thinks. Measured rather than argued: with `affine.py`
mutated so the refinement discharges **every** obligation it evaluates,
`x0 ∈ [0.0, 1.0] ⊢ x0 >= 2.0` is still REFUTED at `refine="affine"`,
byte-identically to `refine=None`, while an interval-undecided obligation flips
to VERIFIED as expected. A property that cannot be made to fail by breaking the
thing it watches is not watching it. The falsifiable half ships instead, as
`test_a_refuted_is_false_at_some_admitted_point`, which runs both refine legs
and forbids a `violated-over-set` over an exactly-empty admitted region — and
whose control is the commit where that actually happened.

**Inserting a box-implied `assume` must not add proving power — DROPPED ON A
FALSE PREMISE, AND RESTORED.** It shipped dropped, with the reason "no commit in
this tree's history and no one-line mutation makes it fail". The first half is
true. The second was written down without being run, and it is wrong. One line
in `propagate._classify_assumed_pred` — the `eq` branch's
`IntervalArray(los=ks, his=ks)` used for *every* comparison, so an assumed
`x >= k` pins `x` to the point `k` instead of meeting a half-space — makes it
fail at once:

| | verdict |
|---|---|
| `x ∈ float64 [-1,1] ⊢ x <= -0.5` | UNKNOWN |
| `x ∈ float64 [-1,1]`, `assume(x >= -1.0)` `⊢ x <= -0.5` | **VERIFIED** |

The inserted `assume` restates the declared lower bound; it cannot add
information; under the mutant it proves the obligation. The property is back as
`test_inserting_a_box_implied_assume_adds_no_proving_power`, with that mutation
registered as the `redundant-assume` control. It is a mutant and therefore
weaker evidence than a commit — but three of this suite's controls were already
invented mutants, so by its own rule it qualifies.

**The lesson is about the rule, not the property.** "No control exists" and "I
did not look for a control" print the same sentence in a docstring, and only one
of them is a reason to drop a property.

Two clauses were also refused *inside* properties that did ship, and the
refusals are recorded next to the clauses that replaced them:

* *"widening a declared bound must not turn VERIFIED into REFUTED"* — unsound.
  Widening STRENGTHENS the claim: `x ∈ (0,1) ⊢ x <= 1` is VERIFIED and
  `x ∈ (-3,4) ⊢ x <= 1` is correctly REFUTED. What holds is the monotonicity
  direction, `UNKNOWN -> VERIFIED`.
* *"reordering independent statements must not change the verdict"* — unsound
  for `assume`/`assert_` transpositions, because narrowing in this tree is
  deliberately forward-scoped, so moving an `assume` across an obligation moves
  a real scoping boundary. What holds is equality for two adjacent `assert_`s,
  and no *contradiction* for anything else over a provably non-empty admitted
  set. Two adjacent `assume`s are in the second bucket, not the first: a single
  narrowing pass does not commute — with `x, y ∈ float64 [0,10]`,
  `assume(x == 5.0); assume(y <= x) ⊢ y <= 5.0` is **VERIFIED** and the same
  three statements with the two assumes transposed are **UNKNOWN**. Both sound;
  one sharper. (`eq` makes `x` a point, after which `y <= x` is no longer
  relational and narrows; the other way round it is dropped first. An earlier
  version of this bullet gave `assume(x >= y); assume(y >= 5)` as the
  counterexample — that one does not reproduce, because a *relational* assume
  is dropped in **both** orders.)

---

## The files

| file | what it is |
|---|---|
| `_grammar.py` | the harness IR, the strategies, the exact oracle, the mutations, and `render()` — which is what makes a shrunk counter-example a hand reproducer |
| `_runner.py` | calling `check`, classifying refusals, and the `Census` that every property asserts a floor on |
| `_profiles.py` | budgets and determinism, one environment variable |
| `_corpus.py` | the deterministic corpus the cross-series differential needs, because two interpreters cannot share a Hypothesis search |
| `positive_controls.py` | where every property is known to fail. Imports nothing |
| `test_oracle.py` | the one-sided oracle, both directions |
| `test_metamorphic.py` | two runs related to each other |
| `test_cvc5_protocol.py` | the cvc5 record protocol as a fuzz target, flat and stateful |
| `test_cross_series.py` | jax 0.10.2 against jax 0.11.0 |
| `test_generator_floor.py` | the strategies still draw what they claim to |
| `test_suite_disclosure.py` | runs with neither hypothesis nor jax, so "examined nothing" is never silent |

## Measured, on this tree

All at `PYTHONPATH=<tree>/src`, `JAX_PLATFORMS=cpu`, x64 forced on by the
modules themselves, python 3.12.3, 24 cores, on a loaded box — load averages
quoted because they are the only thing that makes a wall-clock figure mean
anything. **Every figure below is anchored to the commit AND the environment
that produced it**, for the reason the collect-only row spells out.

At **`35f9480`**, in a venv with `hypothesis`, `cvc5` and `z3` and *without*
`maddening` or `jaxfluids`:

| | jax 0.11.0 | jax 0.10.2 |
|---|---|---|
| property suite, `ci` profile | 30 passed, 1 skipped, 1 xfailed | 30 passed, 1 skipped, 1 xfailed |
| wall | 66.11 s (load 2.51) | 69.28 s (load 2.82) |
| whole `tests/` | 2470 passed, 13 skipped, 1 xfailed, 228 s | — |
| `--collect-only` ids | 2483 | 2483 — **0 lines of diff** |

and on the **shared jax venvs** — `cvc5`, `z3`, `maddening`, `jaxfluids`, and
**no `hypothesis`**, which is also what CI installs:

| | jax 0.11.0 | jax 0.10.2 |
|---|---|---|
| whole `tests/` | 2476 passed, 7 skipped, 155.01 s, `verdict=made` | — |
| `--collect-only` ids | 2478 | 2478 — **0 lines of diff** |

**The collect-only count is a fact about an environment, not about a tree, and
it was recorded here as though it were the second.** This table used to carry a
bare `2476`. That number was right — for the shared venvs, at `9cefc6d`. In a
venv carrying `hypothesis` instead of `maddening`/`jaxfluids` the same commit
collects **2453**; add `cvc5` and `z3` to that venv and it collects **2480**.
Three different true answers for one commit, and the row said none of which it
was. So: the commit and the installed optional dependencies are now part of
every figure.

**The part that cannot rot is the comparison, and it is the part worth having**
— *same commit, same environment, two jax series: 0 lines of `--collect-only`
diff*, in both environments above. That claim survives anyone installing
anything, and it is the claim the two-series support actually rests on.

The one skip in the property suite is the cross-series differential, which
needs a second interpreter; run with one, it passes in **both** directions
(0.11 driving 0.10, and 0.10 driving 0.11).

**The `ci` profile is deterministic.** Three consecutive runs gave
byte-identical outcomes, which is what `derandomize=True` buys and why a
per-push gate uses it: this suite either fails on every push or on none, and
never reddens an unrelated PR a quarter of the time.

**~66 s per push, not the ~6 s an earlier estimate quoted, and the difference
is not overhead.** That estimate priced two properties (an integer oracle at
500 examples and the cvc5 state machine at 1000x10). Those two still cost
about that here — 6.8 s and 1.1 s. The rest is seven more properties, five of
which call `check()` **twice per example** because that is what a metamorphic
property is, plus the 1000-example budgets on `conjunct` and `widen` argued
above (+21 s, and they buy two blatant unsoundnesses the 250-example budget
could not see). `STELLING_PROPERTY_SCALE=0.5` halves the lot if a future job
needs the room; the census floors are what will complain first.

**The randomised `dev` profile found two defects — in this suite, not in
stelling.** Both were invisible at the `ci` budget and both are now fixed:

* `wrappable_constants()` — the MASK the oracle properties use — RAISED rather
  than answering on any constant subexpression the exact integer folder did not
  know. `x <= (0.0 / 0.0)` beside an `int8` declaration was enough. Three runs
  out of three at 1000 randomised examples; zero at 250 derandomised ones.
* `widened()` clamped the widened bound to the dtype range, so a declaration
  whose bound was *already* outside it — `any_array((), "uint8", (-1, 0))`,
  which stelling accepts — had its `lo` moved UP and the "wide" box was a
  strict SUBSET. The monotonicity property reported `UNKNOWN -> VERIFIED` and
  was right about the two runs it was handed. One run in three.

That is the argument for the nightly profile in one paragraph, and it is the
suite's own lesson turned on itself: **a green run at one budget says nothing
about another.** Five `dev` runs are green after the fixes.

## Measured facts worth not re-deriving

* **Shrinking on the harness grammar is the one mechanism that justified the
  dependency.** Median 3 AST nodes over 25 seeds, machine-reduced, no hand
  editing. Everything else Hypothesis adds over a 200-line hand-rolled driver
  was measured and found small here.
* **`hypothesis.extra.numpy` is complementary but structurally blind to this
  project's own defect class.** `from_dtype` is dtype-faithful by construction,
  so it produced **0 of 3000** bounds outside their dtype's range. It uniquely
  adds NaN endpoints and all four unsigned widths; it must not become the only
  source of bounds.
* **`target()` was counterproductive** on the one property it was A/B'd on:
  21/25 seeds with it against 25/25 without, median 169 examples against 115.
  It is used nowhere here.
* **`derandomize=True` FORBIDS the example database** (`InvalidArgument`). You
  get determinism or a replaying corpus, never both — which is why `ci` and
  `nightly` are different profiles rather than different flags on one.
* **The example database buys nothing in ephemeral CI**, measured: the same
  failure at the same example count with and without it.
* **Stateful testing reached both cvc5 defects at ~12x the flat fuzzer's
  cost** (8165 examples against 673), because the driver's record ORDER is
  fixed and a template that hard-codes it spends its whole budget on the slots.
  Both legs ship; the flat one is the cheap one.
