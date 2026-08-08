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

One command, and it refuses by resolved path to install into the shared venvs:

```
tools/property_venv.sh                # jax 0.11.0, ./.venv-prop-0.11.0
tools/property_venv.sh 0.10.2         # jax 0.10.2, ./.venv-prop-0.10.2
tools/property_venv.sh 0.11.0 /some/where
```

It reads the `hypothesis` requirement out of `pyproject.toml`'s dev group, so
the version lives in one place. `stelling` itself is **not** installed into the
venv — the suite is driven with `PYTHONPATH=<tree>/src`, which is what lets one
venv be pointed at any worktree.

## Running

```
# against this tree
JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src .venv-prop-0.11.0/bin/python -m pytest -ra tests/property

# against SOME OTHER worktree or revision — the properties come from HERE,
# the code under test comes from THERE
python tools/property_check.py --tree /path/to/someone-elses/worktree
python tools/property_check.py --rev  fb34e0d

# demonstrate every positive control (each must FAIL where it is supposed to)
python tools/property_check.py --controls \
    --other-python /path/to/venv-with-the-other-jax-series

# the cross-series differential needs two interpreters
STELLING_PROPERTY_OTHER_PYTHON=/path/to/venv-jax-0.10/bin/python \
  PYTHONPATH=$PWD/src .venv-prop-0.11.0/bin/python -m pytest -ra \
  tests/property/test_cross_series.py
```

Budgets are chosen by `STELLING_PROPERTY_PROFILE`:

| profile | seeds | budget | database | what it is for |
|---|---|---|---|---|
| `ci` (default) | derandomized | 1x | none — forbidden alongside `derandomize` | the per-push job |
| `dev` | random | 4x | `.hypothesis/` | working on a property |
| `nightly` | random | 40x | `STELLING_PROPERTY_DB` if set | a scheduled sweep |

`STELLING_PROPERTY_SCALE` multiplies any of them.

**There is no nightly job.** Adding one needs a `schedule:` trigger in
`.github/workflows/ci.yml`, and that file was being edited concurrently when
this suite landed, so the change was kept to a single appended job. Until a
trigger exists, the nightly recipe is a command somebody has to type:

```
STELLING_PROPERTY_PROFILE=nightly STELLING_PROPERTY_DB=/some/cached/dir \
  PYTHONPATH=$PWD/src .venv-prop-0.11.0/bin/python -m pytest -ra tests/property
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
finding the version that holds. Three in this suite were rewritten and two were
refused outright; each docstring records the refusal and its counterexample
next to the clause that replaced it. See "Refusals" below.

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
   be demonstrated does not ship. That is a rule, not a preference; two
   properties were dropped from this suite under it.

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

Two properties are **not here**, and the reasons are the useful part.

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

**Inserting a box-implied `assume` must not add proving power.** Sound, and no
commit in this tree's history and no one-line mutation makes it fail. Dropped
under the rule above. The defect it would plausibly have caught (a vacuous
predicate narrowing a sibling) needs the vacuous predicate *conjoined onto* an
existing narrowing assume rather than inserted beside one, and
`test_a_conjunct_that_adds_no_information_adds_no_proving_power` covers that.

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
  narrowing pass does not commute — with `x, y ∈ [0,10]`,
  `assume(x >= y); assume(y >= 5)` leaves `x ∈ [0,10]` while the other order
  narrows `x` to `[5,10]`. Both sound; one sharper.

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
