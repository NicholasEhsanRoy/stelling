# The eager construction-site detector — why it raises, why it raises `BaseException`, and why there is no carve-out

**Status:** architecture decision, 2026-08-19. Every number below was measured
on jax 0.11.0 and jax 0.10.2 unless a line says otherwise. The de-risking spike
that produced most of them is `spike/B16-derisk`; the code it justified is
`src/stelling/_tripwire/eager.py`, the second hook in
`src/stelling/_tripwire/_adapter_jax.py`, and
`tests/test_tripwire_eager.py`.

## The door

The overflow tripwire hooks jax's const-fold rule for `convert_element_type`.
That catches the INLINE door: `x + 256` on an `int8` array traces to
`add a 0:i8[]`, the rule is handed the written `256`, and the tool sees it die.

It does not catch the EAGER door. `jnp.full((), 256, jnp.int8)` is `0` before
any primitive is bound. The value is narrowed inside
`jax._src.lax.lax._convert_element_type`:

```
if type(operand) is int and new_dtype != dtypes.float0:
  arr = np.asarray(operand).astype(new_dtype)
```

Nothing downstream can tell that the `0` in the jaxpr was written as a `256`.
`SOUNDNESS.md`'s integer-literal wrap entry is exactly this defect and its cost
is a **wrong VERIFIED**: a harness whose obligation is false at all eleven
declared points, certified.

`tests/test_tripwire_gate_coverage.py::GATE_COVERAGE` enumerated 32
construction routes — 17 watched, 7 unwatched, 3 loud, 5 deferred. **Six of the
seven unwatched routes narrow at that one line**, each reaching it with the
written value intact, on both series. The seventh, `np.asarray(N).astype(dt)`,
never reaches jax at all.

## Why a hook on a private function, and not something public

Measured, not assumed. `jnp.full is not jax.lax.full`, and patching both of
those public names gets **0 hits** on `jnp.full_like` and on
`jnp.stack`-of-`full`. Every caller inside jax reaches the narrowing by MODULE
ATTRIBUTE — there are zero `from … import _convert_element_type` in the
installed tree — so one attribute patch covers all six routes and nothing
public covers any of them completely.

That is the same shape of finding the const-fold hook already rests on, one
layer over, and it is why the second hook lives in `_adapter_jax.py` rather
than in a new module: `design/private-jax-boundary.md`'s exemption is pinned to
that exact path by two independent controls. The exemption bought *one file may
name a private jax module*, not *one private jax module may be named*.

## Two modes; this is the second

**Mode 1** would record the eager narrowing and refuse the verdict from inside
`preconditions.check()`'s trace gate. It needs ATTRIBUTION: some way to decide
that the constant destroyed at 11:04 is the constant the trace at 11:05 is
standing on. The spike drove four attribution strategies (global, window,
identity taint, value taint) and none is built here.

**Mode 2** needs no attribution at all, because it never has to connect two
events. It raises at the construction site, in the frame that wrote the
constant, while that frame is still on the stack. That is the whole argument
for building the second one first: it is the one whose correctness does not
depend on a heuristic.

## Why the exception inherits from `BaseException`

The alarm fires inside arbitrary user code. Numerical Python is full of
`except Exception:` — retry loops, fallback kernels, "try the fast path",
warnings-to-error shims — and this repository's own guardrails catch
`Exception` on principle so that an instrument can never break the thing it
measures. An alarm that says *the constant you wrote does not exist in the
program that will run*, swallowed by a handler written for a different purpose,
is a silent program with extra steps. `KeyboardInterrupt` and `SystemExit`
inherit from `BaseException` for the same reason: they are not the running
program's errors to handle.

**The claim is narrow and is stated in the code.** "Uncatchable" is not
achievable in Python. `except BaseException:`, a bare `except:` and
`contextlib.suppress(BaseException)` all still catch this. The claim is exactly
that the COMMON swallow does not.

**The blast radius was measured before the choice was made**, and re-derived
here rather than carried over from the spike.

*Imports.* 25 modules imported with the detector armed — flax, optax, jax_md,
diffrax, lineax, chex, equinox, jraph, e3nn_jax, maddening, scipy (+
`scipy.optimize`, `scipy.linalg`), matplotlib (+ `pyplot`), PIL (+ `Image`),
h5py, jaxlib, jaxfluids, and five jax subpackages: **29 scalar integer
conversions, 0 truncations, 0 fires.**

**Every figure in this section was taken with `jit` on**, which is the default
and is not the only configuration users run; the section below on the origin
filter is what happens with it off, and is the reason that qualifier is now
written down instead of assumed.

*Re-measured for the origin filter, on jax 0.11.0 and 0.10.2, and this is the census the
table below refers to:* **122 module imports** covering 64 third-party
top-level packages — 174 scalar integer conversions, 0 truncations, 0 fires —
and **33 real workloads** across 24 of those packages — 264 conversions and
exactly 1 truncation, which is a control of this repository's own that must
fire and does. Every one of those figures is identical on 0.10.2.

*Execution, which is the stronger half and which the spike did not do.*
Eleven real workloads across eight of those libraries — an optax Adam step, a
flax `nn.Dense` init-and-apply, a diffrax `diffeqsolve`, an equinox MLP under
`vmap`, a lineax `linear_solve`, a jax_md periodic space map, a jraph
`GraphNetwork`, `jax.random` including `randint`/`bits`/`permutation`,
`scipy.linalg.solve` beside a `numpy.astype`, `jax.scipy.linalg.solve`, and
shift/mask/packbits on `uint8`: **65 scalar integer conversions, 0
truncations, 0 fires.**

*This repository's own suite.* 19 fires in 5424 int→int scalar eager
conversions (0.35%) at 9 source lines in 2 files, **0 in library code and 0 in
third-party code** — the spike's figure, and this batch's own run of the suite
with the detector armed session-wide reaches the same set of sites, each now
carrying a region declaration naming why the truncation is the subject of the
code.

So the radius in which the choice matters at all is small, and every site
inside it is one this repository owns.

**IT NEEDS AN ORIGIN FILTER, AND THIS PARAGRAPH SAID IT DID NOT.** What stood
here was: the const-fold tripwire fires on jax's OWN constants —
`jax.random.key(0)` folds `4294967295 -> -1` inside `threefry2x32.py` — and
carries `record.attribute`, an `ORIGIN_JAX` bucket and a suppressed-findings
report for it, while the eager detector never sees it and therefore needs none
of that; *"an instrument that raised inside jax's own PRNG would be unusable at
any blast radius."*

**The premise is true with `jit` on and false in a mode users deliberately turn
on**, and every measurement above was taken with `jit` on. Under
`jax.disable_jit()` jax evaluates `jnp.bitwise_and(seed, np.uint32(0xFFFFFFFF))`
eagerly; the mask reaches this hook as a written scalar; and the version without
a filter raised `EagerTruncationError(4294967295 -> -1, int32)` **inside jax's
own PRNG**. Measured on jax 0.11.0 and 0.10.2:

| configuration | fires, 33-workload census | what fired |
| --- | --- | --- |
| `jit` on, before and after | 1 | the control that must fire |
| `JAX_DISABLE_JIT=1`, before | 9 | the control, plus **8 in jax's PRNG**: `jax.random`, flax linen, flax nnx, equinox, diffrax, jax_md, e3nn_jax |
| `JAX_DISABLE_JIT=1`, after | 2 | the control, plus the one disclosed coincidence below |

`jax.disable_jit()` is jax's own documented debugging workflow and is what the
public `chex.fake_jit()` and `chex.fake_pmap_and_jit()` install: chex's own
installed suite went from **2 failed, 32 passed** to **34 passed** with the
detector armed over `chex/_src/fake_test.py`. And the message prescribed an
impossible remedy — the user never wrote `4294967295`, for flax nnx the line it
named was inside `flax/nnx/rnglib.py`, and `except Exception:` could not
contain it. The only escapes were wrapping every `jax.random` call in
`expected_truncation`, or not arming.

The two instruments were behaving OPPOSITELY in the same environment: with
`JAX_DISABLE_JIT=1` and both armed, the const-fold tripwire reports `NOT ARMED
[not-invoked]` and disables itself, while the eager detector reported `armed`
and failed every test that touched `jax.random`. The older instrument failed
safe there; the newer one fired on jax.

**The fix is the origin question, not a `disable_jit` special case.**
`record.attribute` already answers *"did the user write this constant, or did
jax?"*, and its docstring already records that "innermost non-jax frame" — which
is exactly what `eager._writer_frame` is — *"would print the user's
`jax.random.key(0)` line and claim they wrote `4294967295`"*. What it cannot
lend is its implementation: it keys on the trace boundary, and there is no trace
here. Measured, there is no frame shape to key on either — `jnp.full((), 256,
jnp.int8)` and `jax.random.key(0)` both present as a user frame with nothing but
jax frames beneath it.

So `eager._origin` asks the same question of the DATA:

> is the narrowed integer among the arguments of the call that crossed out of
> non-jax code into jax?

`256` crossed as `full`'s `fill_value`; `4294967295` was manufactured nine
frames below `random.key` and is in nothing the caller handed over. **A call
boundary exists whether or not a trace is in progress, which is why this answer
does not depend on `jit`** — driven as an equality in
`test_the_origin_answer_does_not_depend_on_a_trace_being_in_progress`, which
runs the same four programs with `jit` on and off and requires the same verdict.

Suppression requires POSITIVE evidence — a jax boundary, and a scan of its
arguments (`record.carries`) that COMPLETED and found nothing. A scan cut short
by its own limits returns `None`, is counted in `eager.INCONCLUSIVE`, and is
attributed to the user: over-reporting is visible to a reader holding the quoted
line and a suppression is not, which is the direction `record.attribute`'s own
fallback leans. Everything suppressed is counted in `eager.SUPPRESSED` and
printed with its site, exactly as the tripwire's `suppressed_jax` is.

**Its two edges lean opposite ways and both are disclosed.** A constant that
reaches jax inside a custom object rather than a plain value, tuple, list or
dict is not found by the scan, so it is attributed to jax and does NOT raise —
the one missed narrowing this rule can produce. And a constant jax wrote that
happens to EQUAL something passed at the same call IS attributed to the caller
and raises: measured in the census, `jax.random.PRNGKey(2**32 - 1)`, where jax's
threefry mask and the seed are both 4294967295. That one is the remaining fire
in the table above; the remedy for it is `expected_truncation`, not
`intentional_wrap`, because the declared value would change the program.

**The cost, stated rather than discovered.** `finally:` blocks still run and
context managers still exit, so ordinary cleanup is unaffected. Cleanup written
as `except Exception: release()` does NOT run, so a caller who releases a
resource there and not in a `finally:` will leak it. pytest reports a
`BaseException` raised in a test BODY as a **FAILURE** — measured, `1 failed`,
not `1 error`; this said ERROR and reasoned from it, and `eager.py`'s docstring
had it right. One raised during COLLECTION or in a fixture is an error, which is
the handling any other exception gets there. The `BaseException` choice changes
what can SWALLOW the alarm, not how pytest files it.

**One consequence worth naming, because it was met while building this.**
`tests/test_tripwire_arm.py::_rejected_under_strict` classifies exceptions with
`except Exception:` and drops anything that is not a `TypePromotionError`. Had
the alarm been an `Exception`, that helper would have gone on returning a
confident answer about a door it never drove. The `BaseException` choice is
what makes the missing declaration visible there.

## Error by default on every truncation, and no value-based carve-out

The obvious refinement is to guess intent from the numbers: let `0xFF` into
`int8` through as a mask idiom and stop `300` into `int8` as an accident. It
cannot be done, and the spike proved it rather than asserting it.

`jnp.full((4,), 0xFF, jnp.int8)` and `jnp.full((4,), 255, jnp.int8)` produce
**identical observations at the hook**: the same written value, the same target
dtype, the same result, the same frame. They differ only in source TEXT.

**And the text IS reachable** — `record.source_line(file, line)` returns it and
`eager._message` calls it three statements later, so "not available at the point
the decision has to be made" was false about this repository's own code and is
withdrawn. The ruling stands on its other leg, which cannot be argued away: a
variable, an imported constant, a computed value and a constant defined in
another module carry no text at the site at all, so a rule that reads the line
works for literals and abstains for everything else — and `jnp.full(shape, MASK,
jnp.int8)` with `MASK = 0xFF` one module over is the mask idiom the rule was
supposed to recognise. Intent is not a function of `(value, dtype, result)`, and
not a function of the line either.

Two candidate rules, driven over a corpus of real narrowings:

| rule | hard-errors correct code | lets a real bug through |
|---|---|---|
| A: a value BELOW THE DTYPE'S MINIMUM is deliberate (for an unsigned dtype, exactly "a negative literal into an unsigned type") | 7 | 1 |
| B: an all-ones result is deliberate | 5 | 2 |

Both rules are wrong in both directions. **Neither shipped.**

**And the corpus contains the proof, not just the scores.** `0xFF` into
`int8` is a mask idiom; `255` into `int8` is a saturated pixel written into a
signed byte. Those are the SAME `(value, dtype)` pair, so every function of
what a hook can observe gives them the same answer and one of the two answers
is wrong. That makes the class of value-based rules EMPTY rather than merely
badly-scoring, which is a much stronger thing to be able to say to the next
person with a cleverer discriminator. The corpus, the collision and both
scores are in `tests/test_tripwire_eager.py` and are recomputed by the suite,
so the two figures above cannot drift from the table they came from.

## The two declarations

### `stelling.intentional_wrap(value, dtype)` — the primary

Returns the wrapped integer, computed in Python integer arithmetic by
`record.narrow` — the same recomputation the tripwire's report already checks
every observed narrowing against, so a declaration and a report cannot
disagree about what a wrap is.

It returns the WRAPPED VALUE rather than marking the original, and that is a
decision with a measurement behind it. jax's narrowing branch is guarded by
`type(operand) is int`, which no `int` subclass satisfies — so a marker object
would have changed which code path jax takes, which is exactly what a
declaration must never do. Returning the wrapped value means there is nothing
left to detect: the program is byte-identical to one an author wrote by typing
the wrapped value themselves.

Three properties follow, and they are why this shape rather than a flag, a
suppression list or a decorator:

* **exact**, because the author asserts it — the same standing `assume()` has
  in a verdict: carried as a premise and disclosed, never inferred;
* **cannot license a different site**: it is a value, not a mode, so it changes
  what happens at the one expression it is written in and nothing else, in any
  other thread or one line later;
* **cannot HIDE A TRUNCATION at a different dtype** — which is narrower than
  the claim that stood here, *"cannot license a different dtype"*, and is the
  one that survives measurement. Sometimes the drift fires:
  `intentional_wrap(0xFF, "int8")` is `-1`, out of range for `uint8`. Often it
  does not: `intentional_wrap(300, "int8")` is `44`, which every other integer
  dtype holds. Over this project's own `WRAP_GRID` against the seven other
  dtypes — 98 (declaration, misuse) pairs — **53 (54%) pass silently** and 45
  fire. What survives is the safety property: in every silent case the value
  written at the new site is in range there, so the site performs no narrowing
  and there is no truncation for the declaration to have hidden. What a drifted
  declaration can still do is write the wrong constant — `44` where `300` was
  meant — and that is a bug this instrument does not claim to catch, in either
  direction.

A declaration is **recorded** — site, count and the arithmetic — and printed in
the session report. A premise nobody can see is indistinguishable from the
silence the tool exists to end.

### `expected_truncation(reason)` — the narrow second

For code whose SUBJECT is the truncation. This repository has several: the
doors in `tests/test_tripwire_arm.py` are driven precisely to demonstrate that
they narrow in silence, and `SOUNDNESS.md`'s reproducer exists to be executed
and observed wrapping. `intentional_wrap` cannot serve those — the point of the
line is that `300` becomes `44`, and a line that writes `44` no longer shows
it. For the fence it cannot serve at all: the reproducer is read out of
`SOUNDNESS.md` and executed verbatim, so there is no source to edit, and
editing it would be editing the defect out of the disclosure.

It is deliberately the awkward one: mandatory reason, and **every truncation it
permits is counted and printed with its site and its reason**. An opt-out that
hid what it suppressed would reintroduce the same silence one level up.

**It is dynamically scoped to one context's region stack and it is NOT
lexically bounded**, which is what this section and `eager.py` used to say in
three places. A `with` block looks lexical; no context manager can be. Measured,
in the three directions it matters:

* **threads: isolated** — a region on one thread licenses nothing on another;
* **asyncio tasks: isolated**, because the stack is a `contextvars.ContextVar`
  and every task runs in its own copy of the context. A `threading.local` — the
  first implementation — licensed a truncation in a SECOND task on the same
  loop, since one thread runs them all;
* **generators: NOT isolated, and this is the residue.** A plain generator
  shares its caller's context (PEP 550/568 was never implemented), so a region
  it entered and has not left is open in whatever code resumes it, until the
  generator runs to completion, is closed, or is collected. Nothing in Python
  fixes this. It is disclosed in `report.EAGER_UNCOVERED`, driven in
  `test_a_region_is_DYNAMICALLY_scoped_and_says_so_in_all_three_directions`,
  and it is one more reason this is the awkward declaration and
  `intentional_wrap` — which has no scope at all beyond the expression it is
  written in — is the primary. It lives in `stelling._tripwire.eager` and not
in the public namespace, because it is not the answer to "the detector is noisy
in my code".

## Fail closed, and what that has to mean for an attribute patch

For the const-fold tripwire, "fail closed" is mostly about the registry
disappearing — which is loud. Here the failure that matters is different and it
is silent: the attribute survives, the wrapper stays installed, and jax stops
routing ONE construction route through it.

So arming does four things and refuses on any of them:

1. **locate** — the module and the attribute are where they are expected
   (`no-site-module`, `no-site`);
2. **signature** — `inspect.signature`'s first two parameters are still
   `operand` and `new_dtype`, in that order, and still passable positionally
   (`signature-drift`). A hook that read the wrong argument would raise about a
   truncation that did not happen, at a line that did not write it;
3. **every route, positively** — all seven construction spellings the detector
   claims, across the two different jax branches (`type(operand) is int` and
   the NumPy-scalar path). One going blind is `route-blind:<route>` and it does
   not attach. Keeping five routes and losing one quietly is not a trade the
   tool makes on a user's behalf;
4. **the negative direction** — an in-range value must not be reported
   (`cries-wolf`), and the array jax actually built must hold the value this
   tool's own arithmetic predicts (`mis-attributed`).

The source hash is RECORDED and never gated on, keyed on the exact release, on
exactly the discipline `_KNOWN_HASHES` documents. The two sites move on
different releases, which is why each has its own map: 0.10.2 and 0.11.0 are
byte-identical at the const-fold rule and differ here; 0.11.0 and 0.11.1 are
the other way round.

## One displacement instrument, covering both hooks

B15's audit found a fourth way the trace gate's watch goes partial, and the
gate consulted none of it. Rebinding the const-fold registry entry over
stelling's wrapper after arming leaves the recorder's identity unchanged and
`fires_count()` unchanged — so both of the gate's existing partiality tests
pass — while the wrapper is never called again, so the fire counter stays at
zero and `check()` returns **VERIFIED on a route the inventory calls
`watched`**. Driven on `main` before this batch: `VERIFIED`. Driven here:
`UNKNOWN — trace NOT FULLY OBSERVED`.

Building an attribute patch meant building a displacement check anyway, so
there is ONE: `_adapter_jax.displacement_check()` answers, for every hook this
process armed, whether stelling's wrapper is still the live one.
`_tripwire.displaced()` is its predicate form and it has three callers — the
trace gate, the pytest plugin's end-of-session revalidation, and the nightly
canary. Two instruments, one per hook, would have been two chances to teach one
caller about one hook and forget the other, which is the shape of the defect
being fixed.

## What it costs, and what it does not close

**Cost:** no measurable change across four full suite runs (8m52s baseline,
8m47s instrumented).

**Closed:** six of seven `unwatched` routes, plus `lax.select`-of-`full` and
`jnp.take`'s `fill_value`, plus `report.UNCOVERED`'s scoped
`with jax.disable_jit():` bullet — that one because the reason the const-fold
rule is handed `-56` instead of `200` is that the constant was narrowed at this
very site on the way in.

**Not closed, and named:** two numpy routes. `np.asarray(N).astype(dt)` is
PERMANENTLY unhookable — `np.ndarray.astype` is an immutable type attribute, so
there is nothing to patch, and numpy emits no warning for it even under
`warnings.simplefilter("error")`. `jnp.asarray(np.array(N), dtype=dt)` is a
second spelling into the same residue. Both are declared `unwatched` in
`GATE_COVERAGE`, `silent` in `EAGER_COVERAGE`, named in `report.UNCOVERED` and
in `report.EAGER_UNCOVERED`, and a test asserts the residue is exactly those
two so a third cannot join them quietly.

**Also not closed:** eager execution of the INLINE door (`a + 256` outside
`jit`, 0 conversions seen at this site), `jnp.where`/`jnp.clip`/`jnp.pad`, and
`x % N` / `x // N` / `searchsorted`. Those reach neither hook; they are
`report.UNCOVERED`'s bullets 1, 2 and 3 and they stay open. (Inside a
`jax.disable_jit()` block they DO reach this hook, which is what the scoped
`disable_jit` bullet above is about; outside one, jax traces them and the
constant never arrives here as a written integer.)

**And the origin filter's own two edges**, which are new with it and lean
opposite ways: a constant reaching jax inside a custom object is attributed to
jax and does not raise (the only missed narrowing this rule produces), and a
constant jax wrote that equals something passed at the same call is attributed
to the caller and does raise (`jax.random.PRNGKey(2**32 - 1)`). Both are in
`report.EAGER_UNCOVERED` and both are driven.

## The numpy fence, abandoned by ruling

`DynamicJaxprTrace.new_const` was measured as a genuine single choke point for
numpy ingestion. It is not usable: a plain Python `x + 0` arrives there as a
numpy `int8` array indistinguishable from a user's, so a narrow-integer fence
would reject every integer literal in every jax program. Recorded here so the
next person does not re-derive it.
