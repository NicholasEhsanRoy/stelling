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
property still has a control, ten controls still fire, and the suite still
runs at all. **That is rot, caught on the push that causes it — not a soundness
argument.**

### Landing this suite withdrew a repo-wide pin

`tests/test_skip_inventory.py::test_no_session_skip_is_undisclosed` claims the
whole suite's skip set is complete. It **withdraws that claim** — by skipping,
with its own reason — on any session that reported an `xfail`, and this suite
ships two:

```
the completeness pin is WITHDRAWN, not passed: this session reported
2 test(s) as xfailed.
```

**IT SAID ONE, AND SHIPPED ONE, UNTIL THE FLOAT ORACLE LANDED.** The quoted
line above read `1 test(s) as xfailed`, and the sentence above it read "this
suite ships one".
`tests/property/test_float_oracle.py::test_the_executed_value_lies_inside_the_computed_box`
is the second: `xfail(strict=True, raises=ExecutedValueOutsideBox)`, for the
open class in which the value the compiled program computes falls outside the
box the propagator computed for it. The count in that sentence is derived by
the pin from `len(XFAILED)`, so nothing had to be edited for the line itself
to stay true — which is exactly why the prose around it had to be, and is the
reason this paragraph exists instead of a silently corrected numeral.

Measured on the whole tree with hypothesis installed: `3910 passed, 13 skipped,
1 xfailed`, exit 0, that pin among the skips — re-driven at `3482822` on jax
0.11.0 with hypothesis 6.165.10, CPython 3.12.3, `JAX_ENABLE_X64` unset. It
read `2470 passed` until then; the skip and xfail counts have not moved, only
the size of the suite, which is why the sentence around them still holds. It
is the pin's own rule
(disclosed ⇒ withdrawn, never failed — the same cut `N deselected` gets), and
it is not a defect in the pin. It is a consequence of this suite that nobody
wrote down, and it is written down here, in `CONTRIBUTING.md` and in the
`property` job's own comment block.

**CI is unaffected**, measured rather than assumed: none of the three whole-suite
lanes installs hypothesis, so every module under `tests/property/` gates at
collection, no xfail is reported, and their `verdict=made` assertion holds. The
pin is off for exactly the sessions that *can* run the property suite. It comes
back the day BOTH remedies land and BOTH markers are deleted — the one in
`test_oracle.py` and the one in `test_float_oracle.py`; narrowing the session
does not bring it back, by the same rule. This sentence named one marker until
the float oracle landed, and one is now not enough.

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
that should agree still disagree. **For that defect — one that lives in the
translation — a metamorphic property is strictly stronger than an execution
oracle.**

**NARROWED TO ITS SUBJECT, AND NOT DELETED.** That sentence used to end
*"Where a defect is in the translation, metamorphic properties are strictly
stronger than an execution oracle"*, stated over the whole class of
translation defects, and it was read here as an argument about execution
oracles in general. It is true of the integer-literal wrap it was written
about, and it is exactly as true of any other defect that destroys the
program before tracing. It says nothing about a defect that lives in the
ANALYSIS of a program the trace represents faithfully — for which an execution
oracle is not weaker but is the only instrument there is, because there is no
second run to relate. `test_float_oracle.py` is that instrument, and the nine
programs it pins are nine places where nothing metamorphic in this suite has
anything to compare.

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

* **integers only — RETRACTED, AND HERE IS WHAT IT SAID.** This bullet read,
  in full:

  > **integers only.** `SOUNDNESS.md` records that real mode judges floats in
  > exact real arithmetic while integers are judged execution-faithfully. A
  > float harness can be correctly VERIFIED in ℝ and violated by IEEE
  > execution — the declared posture, not a defect. **An oracle pointed at
  > floats measures the documentation.**

  Every clause of that is true except the last one, and the last one does not
  follow. What the ℝ posture buys is that such a verdict is **disclosed**; it
  does not make the verdict harmless, and it does not make an instrument
  pointed at it an instrument pointed at prose. A VERIFIED that the compiled
  program contradicts at a dtype-representable point of its own declared box
  is unacceptable whatever ℝ says.

  Measured after the argument was withdrawn and the instrument built
  (`test_float_oracle.py`, 2026-08-28 at `874d8ba`, 1500 examples at
  `STELLING_PROPERTY_SCALE=12.5`): **563 violations of box containment — 316
  NaN, 175 flush-or-subnormal, 29 narrow-format rounding, 27 reduction
  reassociation, 16 overflow-to-inf, 0 unexplained and 0 unclassified — of which
  95 are an obligation whose box says "definitely true for all elements" whose
  predicate executes FALSE at an admitted point.** Nine programs are pinned as
  regression cases; eight of the nine violate box containment against
  `v0.1.0`'s own `src/` as well, and five of those falsify a discharge there.
  None of them needed a float harness to be *mis*-documented to be a defect. The scope of THIS
  file's oracle (`test_oracle.py`) is still integers, and that is now a
  statement about the exact-oracle machinery — `_grammar.eval_pred_exact`
  works in unbounded Python integers — rather than an argument that floats
  should not be checked. They are checked, next door.
* **the box must be enumerable.** `_grammar.declared_points` returns `None`
  rather than a partial answer when the product of the declared boxes exceeds
  4096, and the caller discards the example. A refusal to answer, never a
  silent "no".

---

## The differential float oracle, and what a lane without it loses

`test_float_oracle.py` asks the one question the rest of the repository does
not: **is the value the compiled program computes inside the box the
propagator computed for it?** That sentence is what every VERIFIED rests on,
and it is false on this tree in nine pinned programs. The machinery is in
`_float_oracle.py`; the two legs are the containment property (xfail-marked,
strict, `raises=ExecutedValueOutsideBox` — the class is open) and the
residual, which classifies every violation and forbids one that is **proved**
to be a box wrong about ℝ.

**It runs the program twice.** The op-by-op walk is the TRACE's granularity,
not the program's: an equation whose operands are all compile-time constants
is executed at runtime, where the backend flushes a subnormal, instead of
being constant-folded in full precision the way the compiled program folds it
(`jax.jit(lambda: jnp.sqrt(5e-324))()` is `2.2227587494850775e-162`;
`jnp.sqrt(jnp.asarray(5e-324))` is `0.0`, and the box contains the first).
Every candidate violation is re-checked against the same jaxpr staged into one
`jax.jit` and declined if the compiled program does not have it — **76 of 639
candidates** in a 1500-example run. That second route copies
`stelling.falsify._whole_program_route`, including the two properties that
make it safe: it is consulted only after a violation, and it can only decline.

**`UNEXPLAINED` is a proof, `unclassified` is a refusal, and they used to be one
answer.** A violation is `UNEXPLAINED` only when the exact real value of the
equation on the values it ran on is computable and is OUTSIDE the box — the
box is wrong about ℝ. Where no exact reading exists (`sqrt`, `exp`, `sign`,
every reduction but `reduce_sum`) the answer is `unclassified` and is censused.
A green residual leg therefore says *no violation was PROVED to be a box wrong
about ℝ*, and the `unclassified` count says how much of the field that proof did
not cover.

**Three things it is blind to, stated here because a reader of a null result
will look here.** They are each a mechanism in `_float_oracle.py`, not a
warning:

1. **⊤ is unfalsifiable, and it is not the only bucket that is.** Every
   decline lands as `[-inf, inf]`, which contains every finite float, so this
   instrument sees nothing exactly where the propagator already declined. The
   census counts boxes in FOUR buckets, because they are unfalsifiable for
   entirely different reasons and only the last is a place a finite value can
   be caught outside its box. Measured over 1500 examples, of **13727** boxes
   read: **2042 ⊤ (15 %), 1697 empty (12 %, a size-0 declaration), 1577
   integer (11 %, never compared — a binary64 box endpoint cannot represent an
   `int64` above 2**53), 8411 compared (61 %).** Both properties floor on the
   last alone. **The integer bucket used to be counted inside "76 %
   falsifiable"** and reached no census, no floor and no disclosure; 39 % of
   the field of view is blind, not 24 %. The pass rate is not a safety signal.
   NaN is the only cause that survives a ⊤, because no box contains a NaN.
2. **The sampler's grid.** `np.float32(1e-20)` is `9.9999997e-21`, *below* a
   box declared `(1e-20, 1e-10)`; sampling there and comparing against a box
   built for `[1e-20, …]` invents violations of the identity, with no
   arithmetic in them. `_float_oracle.snap_inward` steps inward with
   `nextafter` in the program's own format, and a declaration with no value of
   its dtype inside its box is reported `unsampleable` and contributes
   nothing — a refusal, not a null result. Driven both ways with the two
   `nextafter` steps deleted on a copy of the tree, 600 draws of the unbiased
   leg: **unsnapped, 246 violations of which 162 (66 %) land on `stelling_any`
   itself; snapped, 148 violations and 0 on `stelling_any`.** Two thirds of
   what an unsnapped instrument finds is its own sampler's rounding — and
   nothing asserted that until an audit said so, because with the snap deleted
   the containment leg still XFAILed and the residual leg failed only
   incidentally. The guard is explicit now, and it is not a blanket ban: a
   `stelling_any` violation under a NARROWING ASSUME is a genuine member
   (`assume-narrows-past-the-program`, the ninth). **The same mistake, one
   dtype over, was live in this file**: integer candidates went through
   `float`, so `any_array((), "int64", (1, 2**63 - 1))` sampled
   `-9223372036854775808` — outside its own box — and this module reported its
   own sampler as a defect in stelling. Integer candidates stay Python
   integers now. (**Reported, not repaired here:**
   the declared endpoints stelling stores are binary64 images and are *not*
   snapped to the program's dtype grid, so a float32 box can have endpoints no
   float32 can take. That is a declaration-layer defect; this instrument only
   works around it.)
3. **What the generator cannot reach.** The unbiased grammar cannot build the
   reassociation class *at all* — `_grammar.SHAPES` tops out at four elements
   and `jnp.sum` only splits into two windows at n ≥ 33 — so that is a proof
   about the generator and not a zero in a table.
   `_float_oracle.cancelling_sum_programs` builds the shape deliberately, and
   needs all three of a degenerate envelope, per-element constants and forced
   cancellation at once. Measured, 200 programs of that construction per size:
   **0/200 at each of n = 16, 30, 31, 32; 18/200 at n = 33; 15/200 at n = 34;
   9/200 at n = 64; 0/200 at n = 128** — and every row is byte-identical on
   jax 0.10.2, so the boundary is a property of the XLA lowering rather than of
   one series. **The n = 128 row is quoted with the rest and is NOT
   explained**: 64 exact pairs may simply re-cancel under whatever split XLA
   picks there. It is a fact about this construction, not about the boundary,
   and the n <= 32 rows are what establish the boundary. This paragraph
   omitted that row and then concluded about the boundary anyway, which is
   picking the rows that agree. It finds that class and nothing else, which is
   why it is a separate strategy from the unbiased one.

**How far back the class goes, re-derived rather than quoted.** With
`git archive v0.1.0 src` on `PYTHONPATH` and v0.1.0's two-trace API
(`harness.trace` plus `jax.make_jaxpr`, since `trace_with_jaxpr` did not exist
yet) standing in for today's one: **nine of the ten pinned programs violate
box containment at v0.1.0 too, and five of them falsify a DISCHARGE there** —
`ftz-subnormal-sum`, `f32-underflow`, `f32-single-multiply`, `f32-exp` and
`assume-narrows-past-the-program`.
The one that does NOT reach v0.1.0 is the reassociation member, and the reason
is worth knowing: v0.1.0's `interval.mul` was not exact, so `x * K` boxed to
`[7.205759403792793e+16, 7.205759403792795e+16]` instead of to a point and the
sum's box came out `[-570.0019531250001, 694.0029296875001]` — wide enough to
contain the executed `9.0`. Tightening the multiply to the exact rational
corner is what made that class visible. The slack was never soundness.

Neither generator reaches `scan`/`while`/`cond`, `vmap`, `grad`, the affine
refinement, the solver legs or `semantics="ieee"`, and integer-dtype outputs
are counted but not compared (a box endpoint is a binary64 float and an
`int64` past 2\*\*53 has no exact binary64 image, so the comparison would report
its own rounding). The wrap class those dtypes carry is `test_oracle.py`'s
subject and is judged there exactly.

**Cost, and how it runs.** 29-43 ms per example: a 1500-example run of the
residual leg takes 44 s at load average 6 and 64 s at load average 7 on a
24-core box, and three runs returned the census below byte for byte.
The cost is dominated by jax dispatch — the op-by-op walk always, and one
`jax.jit` compile for each program whose first route found a violation. At the
`ci` profile the whole module costs **14.0-14.9 s**, three runs at load
average 9 (`pytest -q -ra tests/property/test_float_oracle.py`, `1 passed,
1 xfailed`, zero warnings) —
the containment leg stops at its first violation and then shrinks, and the
residual leg runs its full 120 examples plus 11 pinned ones.

**The nearest thing in `src/`, and what it does with these eight.**
`stelling.falsify.probe` (`check(..., falsify="sample")`) also executes the
program at concrete points — but it asks whether the OBLIGATION is false
there, not whether each value is inside its box, and it is default-off and
`experimental`. Driven over all eight members plus the overflow probe,
2026-08-28: **it reports none of the nine.** Two are UNKNOWN, so the probe
never runs (it fires only after a VERIFIED); one is REFUTED; six are VERIFIED
and stay VERIFIED. For four of those six it SAW the executed violation and
declined it, in its own words:

```
falsification probe: 28 point(s) executed … declined 28 float-rounding-artefact.
28 EXECUTED VIOLATION(S) WERE DECLINED, NOT ABSENT: at those points the
obligation evaluated FALSE and the probe would not report it
(28 exact-replay-holds-over-the-rationals).
```

— 28 points for `ftz-subnormal-sum`, 12 for `f32-underflow`, 2 for
`f32-single-multiply`, and 58 for `f32-exp` under
`no-exact-reading-of-this-program`. That decline rule *is* the ℝ defence in
instrument form, disclosed rather than hidden, and it is why the probe is not
a stand-in for this oracle. The other two VERIFIED members produce no
obligation violation at all: `1/Σ(…)` executes `inf > 0` and the float32
square executes `inf > 0`, both TRUE, so the obligation holds while the box
for the equation upstream of it excludes the value that was computed. **A
box-containment oracle sees a defect one equation before an obligation oracle
can.**

**What a lane without hypothesis loses, and what stands in for it.**
`hypothesis` is a dev-group dependency and is installed by **none** of the
three whole-suite lanes, so on the jax, jax-0.10 and zero-dep lanes this module
gates at collection like every other file here, and those lanes check box
containment **not at all**. What stands in
for it in a merge lane today is: the `property` job in `.github/workflows/ci.yml`
(which installs hypothesis and runs both legs on every push, and whose xfail
step asserts **exactly two** `XFAIL` lines so that a marker cannot be weakened
silently), and `tools/property_check.py --control float-oracle-unexplained`,
which is in that job's per-push control list. **There is no stand-in that runs
in a lane without hypothesis**, and that is a gap, not an arrangement: a
zero-dep or jax-lane green tick says nothing whatever about whether an executed
value is inside its box.

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

Add an entry to `positive_controls.py`. Every field below is a placeholder,
the nodeid included — which is why it is spelled as one rather than as a
real-looking nodeid naming a test that does not exist:

```
Control(
    name="my-property",
    nodeid="tests/property/test_<your_module>.py::test_<your_property>",
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

**And one of those two demonstrates its clause only at the `ci` profile.** A
short-circuiting oracle reports a failing example against ONE clause — the
first that fails — so a tree violating two clauses demonstrates whichever the
search reaches first, and that is a property of the example sequence and not
of the tree. `8ef8f75` is such a tree: counted independently over 1500 draws,
**458 examples violate clause (1) and 284 violate clause (2)**, `_judge` tests
(2) first, and which one comes first in the sequence decides what is reported.
At the derandomized `ci` profile it is a clause-(1) example, reproducibly —
`cvc5-exit-tell` fires at scale ×1, ×2 and ×4. At the RANDOMISED `dev` profile
it is **either, run to run**. With `.hypothesis/` removed before every run:

| `--profile dev` | FIRED | NOT DEMONSTRATED | runs |
|---|---|---|---|
| ×1 | 39 | 21 | 60 |
| ×2 | 23 | 17 | 40 |

and every refusal is the clause-(2) failure, of which this is one — the
answer is drawn data, so `'sat'` below varies run to run:

```
FIRED, but the failure did not carry 'ACCEPTED A NONZERO-EXIT RUN'
what pytest recorded: AssertionError: ACCEPTED A TRUNCATED RUN as 'sat' [flat]
== 0/1 controls fired
```

**That table replaces "three runs out of three at ×1 and again at ×2", which
was a replay artefact and is withdrawn.** `dev` is `derandomize=False` and
`_profiles.py` attaches a database only when `STELLING_PROPERTY_DB` is set,
which `property_check.py` pops out of the environment — so hypothesis's default
database at `<repo>/.hypothesis/examples` is live and shared across `--control`
invocations. Driven, 16 chains of four runs with the database wiped only at the
head of each chain: after a first run that reported clause (1), **33 of 33**
follow-ups reported clause (1); after one that reported clause (2), **15 of 15**
reported clause (2). Three consecutive runs are one draw and two replays.

**If you are measuring a control at a randomised profile, wipe `.hypothesis/`
between runs or you are measuring your own first answer.** That is the general
lesson and it cost this entry two rounds.

The direction is still safe — a refusal, never a false green — but
`python tools/property_check.py --controls --profile dev` is an invocation this
file documents, and on roughly a third to two fifths of independent runs it
reports a control NOT DEMONSTRATED. See `_judge`'s docstring for why the
obvious fix (pin the transcript as an `@example`) is not taken here.

So: if your oracle is a conjunction, say in the control's `why` which conjunct
its tree exercises, and register a control for each of the rest or write down
that you did not. **If the tree violates more than one conjunct, say which
profile the demonstration holds at**, because "the control fires" is then a
claim about the sequence too. The measurements for this one are in `_judge`'s
docstring in `test_cvc5_protocol.py`.

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
| `_float_oracle.py` | the differential float oracle's machinery: the inward-snapping sampler, the op-by-op jaxpr interpreter, the box reader, the cause classifier, and the pinned members |
| `_runner.py` | calling `check`, classifying refusals, and the `Census` that every property asserts a floor on |
| `_profiles.py` | budgets and determinism, one environment variable |
| `_corpus.py` | the deterministic corpus the cross-series differential needs, because two interpreters cannot share a Hypothesis search |
| `positive_controls.py` | where every property is known to fail. Imports nothing |
| `test_oracle.py` | the one-sided oracle, both directions |
| `test_float_oracle.py` | is the executed value inside the box the propagator computed? Two legs: the containment property (xfail-marked, open) and the residual, which forbids a violation with no IEEE cause |
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
