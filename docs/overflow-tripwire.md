<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# The overflow tripwire — one line in `conftest.py`

<!-- doc-example: illustrative -->
```python
# conftest.py
pytest_plugins = ["stelling.overflow"]
```

Run your existing suite. That is the whole setup.

Or, without editing anything:

```console
$ pytest -p stelling.overflow
$ pytest --stelling-overflow=auto     # the same thing, as a flag
```

(The flag on its own needs the plugin to have been autoloaded. If your CI sets
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, name the module too — see
[below](#if-your-ci-sets-pytest_disable_plugin_autoload1).)

## What it finds

An out-of-dtype-range Python integer constant is **silently narrowed** on its
way into a JAX trace. On an `int8` array, `x + 256` reaches the jaxpr as

```
add a 0:i8[]
```

The `256` you wrote is destroyed. Reproduce it yourself (verified on JAX
0.10.2 and 0.11.0 — the output is byte-identical on both):

<!-- doc-example: illustrative -->
```python
import jax
import jax.numpy as jnp

def add_offset(x):
    return x + 256

jaxpr = jax.make_jaxpr(add_offset)(jnp.zeros(1, jnp.int8))
print(jaxpr)
# { lambda ; a:i8[1]. let b:i8[1] = add a 0:i8[] in (b,) }
#                                          ^^^^
#                               256 is gone — JAX traced 0

result = jax.jit(add_offset)(jnp.int8([100, 50, -10]))
print(result.tolist())
# [100, 50, -10]  — the function is x + 0 = x, not x + 256
```

No error, no warning, no `RuntimeWarning`
you could turn into one — six supported mechanisms were measured against it
(`numpy_dtype_promotion('strict')`, `enable_checks`, `debug_nans`,
`debug_infs`, `np.errstate(over='raise')`, `warnings.simplefilter('error')`)
and all six leave it silent. `checkify.all_checks` returns `None` on it while
its out-of-bounds and divide-by-zero controls throw.

`jax.numpy_dtype_promotion("strict")` is worth singling out, because it looks
like it should help and does not. Measured over an 11-door grid on both tested
series with x64 on and off — the raise-or-wrap pattern identical in all four
cells — for a *concrete*-dtype operand (`np.int64(256)`, `np.int32(256)`,
`jnp.int32(256)`, `jnp.array(256, jnp.int32)`, `np.bool_(True)`):

A bare Python `bool` is **not** one of these, and this list said `True` until it
was measured: `np.bool_(True)` is rejected at all six promoting doors, but
`x.at[0].set(True)` is accepted silently under strict — five of six, not six.
The operand spelling had been carried over verbatim from a sentence that was
retracted for being wider than its measurement.

| | strict promotion |
|---|---|
| the **6** doors that promote an operand against an array — `x + N`, `x >= N`, `x.at[i].set(N)`, `jnp.where`, `jnp.clip`, `jnp.maximum` | raises `TypePromotionError` |
| the **5** construction doors — `jnp.array`, `jnp.asarray`, `jnp.int8`, `jnp.full`, `jnp.full_like` | **silent**, and the operand narrows |
| the same doors with the *in-range* `np.int64(3)` | raises at the same 6 |
| a bare Python `int`, at any of the 11 | never raises |
| a weakly-typed `jax.Array` (`jnp.asarray(256)`), at any of the 11 | never raises, and it wraps |

So it separates **dtypes, not values**: it flags in-range code and misses the
spellings that actually lose the value. Nor is a rejection a sign the value was
safe — `x.at[0].set(np.int64(256))` raises under strict and wraps to `0` under
standard, so that operand was losing its value too. Adopt strict promotion for
its own reasons; it is not a weaker form of this check.

What does help, one constant at a time, is **hoisting the literal to its own
definition site**: `jnp.array(256, jnp.int8)` and `jnp.asarray(256, jnp.int8)`
both raise `OverflowError` for a Python int, in all four measured cells.

The tripwire attaches to the site where the value actually dies and reports
each one like this:

```
[1] model/quantize.py:41 in to_int8
    41 | return (x - 128) * 300
    RULE      attribution: the innermost frame OUTSIDE JAX inside the traced
              region -- your own code, or a library you called that is not
              jax. Not the entry point, and not jax's caller
    OBSERVED  the constant written there is 300; int8 holds that as 44. Both
              halves read at the site: the rule received 300 (int64) and
              returned 44 (int8).
    ARITHMETIC 300 mod 2**8 = 44, which is < 2**7, so 44
    CONFIRMED recomputed from (300, int8) without the hook: 44. Agrees with
              what ran, so this is reported.
    REPRODUCE
        import jax, jax.numpy as jnp
        print(jax.make_jaxpr(lambda a: a + 300)(jnp.zeros((), jnp.int8)))
        # the 300 is not in the jaxpr: it prints 44:i8[]
    INFERENCE (not observed -- these are suggestions, not claims)
        ...
```

Every finding is **recomputed by an independent route before it is printed** —
the narrowing is redone in plain Python integer arithmetic from the recorded
`(value, dtype)`, without going near the hook — and if the two disagree, the
report prints the *disagreement* and withholds the finding. That is the same
discipline a refuted verdict's witness gets, and it is here for the same
reason: a finding you cannot reproduce costs more trust than ten real ones
earn.

## What it does NOT find

Printed on every run, findings or not, because "no findings" and "your code is
clean" are not the same sentence and this tool will never print the second one.
(In the terminal report it comes LAST, under *what this run did NOT look at*.
This section used to say "stated first", which it is not here — it is the second
section — and is not there either.)

**This table is a floor, not a census** — and read it as the answer to "what
does it not see", because that is what it is for. Every row was measured with a
live control in the same process, on both tested series with x64 on and off,
and **the value wraps in every UNCOVERED row**.

| door | status | measured |
|---|---|---|
| `x + N`, `x * N`, `x >= N`, `x.at[i].set(N)`, `jnp.maximum(x, N)`, `jnp.minimum(x, N)` under a trace | **covered** | fires |
| **eager execution** (outside `jit`) | **UNCOVERED** | 0 invocations |
| **`jnp.where(pred, N, x)`** | **UNCOVERED** | 0 invocations |
| **`jnp.clip(x, lo, N)`** *and* **`jnp.clip(x, N, None)`** — *either* bound | **UNCOVERED** | 0 invocations |
| **`jnp.pad(x, k, constant_values=N)`** | **UNCOVERED** | 0 invocations |
| **`jnp.take(x, i, mode='fill', fill_value=N)`** | **UNCOVERED** | 0 fires (3 in-range visits counted) |
| **`jnp.full(shape, N, dt)`**, **`jnp.full_like(x, N)`** | **UNCOVERED** | 0 fires; the rule sees the already-wrapped value and **counts it in the denominator** |
| **`lax.full(shape, N, dt)`**, **`lax.full_like(x, N)`** | **UNCOVERED** | as above |
| **`lax.convert_element_type(N, dt)`** | **UNCOVERED** | as above |
| **`lax.select(p, jnp.full(shape, N, dt), x)`**, **`jnp.stack([x, jnp.full(shape, N, dt)])`** and anything else built on `full` | **UNCOVERED** | as above |
| **`np.asarray(N).astype(dt)`** — numpy narrows it before jax sees it | **UNCOVERED** | 0 invocations |
| an operand that was already an array | **UNCOVERED** | the fold declines non-scalars, so the wrap already happened |
| **inside `with jax.disable_jit():`** | **UNCOVERED** | 0 fires on a jaxpr *byte-identical* to one that fires outside the block |
| anything replayed from a **warm trace cache** — a `@jax.jit` function any earlier trace already reached, whether before this tripwire armed or after it | **UNCOVERED here, COVERED inside `check()`** | jax's cache is keyed on the jitted callable and its avals, so the body is never traced again and the rule never runs over it. `jax.jit(f, inline=True)` does this leaving no `pjit` in the jaxpr to notice it by. `preconditions.check()` and `contracts.check_contract()` empty jax's trace caches before the trace they gate, so a **verdict**'s observation is complete *with respect to jax's caches, in a single-threaded process* — the next two rows are what that qualifier leaves out. This **session report** has no such moment and watches whatever your suite happens to trace |
| a value narrowed into a **memo jax does not own** and replayed from there — `jax.extend.core.jaxpr_as_fun(saved_jaxpr)`, a user `functools.lru_cache`, **`jax.closure_convert`** | **UNCOVERED, and the `check()` eviction does not reach it** | `jax.clear_caches()` empties JAX's caches and nothing else. Measured on jax 0.11.0, all three: VERIFIED, 0 fires, executed values `[-25536, -25436]` where the source says `[40000, 40100]`. `closure_convert` is a public jax API and traces at *setup*, hoisting the already-narrowed constant into the consts it returns |
| a gated `check()` racing **another thread that traces** | **UNCOVERED, and the `check()` eviction does not reach it** | jax's trace cache is process-global; this gate's fire counter is per-thread. The window between the eviction and the trace it protects is not atomic, and fires caused by another thread are counted on that thread. Measured over 400 gated checks of one harness whose narrowing sits in a shared jitted helper: **0/400** wrong VERIFIED single-threaded, **247/400 (61.8%)** with four threads calling that helper meanwhile (399/400 before the eviction existed). The rate scales with how much the harness traces before it reaches the helper — 1/100 if it is traced first, 52/100 after 50 primitives, 247/400 after 100, 100/100 after 200 — so treat it as a range, not a constant. stelling makes no thread-safety claim: run gated checks on one thread |

There are three distinct causes below, and the second is the one worth knowing
about.

**The site is never reached.** `where`, `clip` at either bound, `pad` and
`take` never fold a constant here: the literal sits at the enclosing call site
and the `convert_element_type` inside the sub-jaxpr operates on a **variable**.
Other mechanisms may reach these; none is built.

**The value is already narrowed before the site.** `jnp.full`, `jnp.full_like`,
`lax.full`, `lax.convert_element_type`, `lax.select`, anything built on `full`,
and a scoped `jax.disable_jit()` all
truncate through numpy first, so the rule is handed a value that is *in range*
and does not fire — **and that visit is counted in the printed denominator.**
So a large denominator is not evidence of coverage. Measured: `x + 300` on
`int8` hands the rule `300`; `jnp.full((), 300, int8)` hands it `44`. Inside
`with jax.disable_jit():`, `a + 200` hands it `-56` where the same line outside
the block hands it `200` — same jaxpr, byte for byte, and no fire.

Process-wide `JAX_DISABLE_JIT=1` is a different case and is handled: `arm()`
reports `not-invoked` and the tool disables itself rather than reporting a quiet
zero. Only the scoped block is silently blind.

Eager execution is uncatchable from Python at all: warm dispatch is eleven
frames of C++ fast path, the constant arrives as a `pjit` argument, and XLA
truncates it.

**The route is never traced at all.** A `@jax.jit` helper whose trace cache is
already warm is replayed, not traced, so nothing runs over its body. This is
the one row above with two different answers, because it has two different
answers: the gate in `preconditions.check()` closes it by emptying jax's trace
caches first, and the session report cannot, because it has no single moment
that owns the whole program.

That eviction reaches exactly as far as jax's own caches, on one thread, and
the last two rows of the table are what lies past it: a value narrowed into a
memo jax does not own is replayed straight through the eviction, and a
competing thread can re-warm a jit body inside the window between the eviction
and the trace it protects. Both are measured in those rows. If you take one
operational rule from this page, take this one: **run gated `check()` calls on
one thread.**

**The enumerated version of this table.** The prose above is a floor and says
so. `tests/test_tripwire_gate_coverage.py` carries the same claim as a
`GATE_COVERAGE` dict — one bucket per construction route — that the suite
*measures* by driving every route through `check()` twice and comparing. A
route that changes bucket goes red there. Two of its buckets are not in this
table at all and are worth knowing: **`loud`**, where jax itself raises rather
than wrapping (`jnp.array(N, dtype=dt)`, `jnp.asarray(N, dtype=dt)`,
`jnp.int16(N)` — note that `jnp.full(shape, N, dt)`, three rows up, silently
wraps the same value), and **`deferred`**, where the written constant reaches
the jaxpr intact and the narrowing is a run-time `convert_element_type` (`x //
N`, `x % N`, `where`, `clip`, `pad`) — the trace gate has nothing to see there,
and the propagation's convert transfer declines the form instead.

## The eager door, and the second instrument that closes it

Everything above is about ONE door: a constant that survives into the trace and
dies in jax's const-fold rule. There is a second door, and the tripwire cannot
see through it at all.

<!-- doc-example: illustrative -->
```python
OFFSET = jnp.full((), 256, jnp.int8)   # this is 0. jax says nothing.
```

The 256 is narrowed at array *construction*, inside
`jax._src.lax.lax._convert_element_type`, before any primitive is bound. There
is no fold to observe, no jaxpr that ever held a 256, and nothing for a
const-fold hook to see — which is why `jnp.full` and everything built on it are
UNCOVERED rows in the table above. `SOUNDNESS.md` records what that costs: a
harness whose obligation is false at every declared point, and a **VERIFIED**.

`--stelling-eager-truncation=error` attaches a second, independent instrument
to that construction site. It does not report; it **raises**, at the line that
wrote the constant:

```console
$ pytest --stelling-eager-truncation=error
```

```
stelling: 300 was TRUNCATED to 44 at its construction site, and 300 does not
exist anywhere in the program jax will run.

    written   300
    dtype     int8  (range -128 .. 127)
    became    44
    arithmetic  300 mod 2**8 = 44, which is < 2**7, so 44
    at        /home/you/project/offsets.py:12, in build()
              return jnp.full((4,), 300, jnp.int8)
```

**It is off by default and it is not turned on by `--stelling-overflow`.** Two
dials, because they are two instruments: the tripwire is a *report* over a
session and can be armed on a suite that has undeclared truncations in it,
while this is a *rule* — a session it is armed on either contains no undeclared
truncation or does not finish.

### It raises `BaseException`, deliberately

`stelling.EagerTruncationError` inherits **directly from `BaseException`**, so
an ordinary `except Exception:` cannot swallow it. That is not defensiveness:
the alarm fires inside arbitrary code — inside `jnp.full`, inside a library's
constructor, inside a `jit` body — and numerical Python is full of
`except Exception:` written for retries, fallback kernels and warnings shims.
An alarm saying *the constant you wrote does not exist in the program that will
run*, caught by a handler written for something else, is a silent program with
extra steps. `KeyboardInterrupt` and `SystemExit` sit under `BaseException` for
the same reason.

**"Uncatchable" is not achievable in Python and stelling does not claim it.**
`except BaseException:`, a bare `except:` and `contextlib.suppress(BaseException)`
all still catch this. The claim is exactly one thing: the *common* swallow does
not. And it has a cost worth knowing before you switch it on — `finally:` blocks
still run, but cleanup written as `except Exception: release()` does **not**, so
a caller who releases a resource there and not in a `finally:` will leak it.

### If the wrap is what you meant, declare it

<!-- doc-example: illustrative -->
```python
from stelling import intentional_wrap

MASK = jnp.full((4,), intentional_wrap(0xFF, "int8"), jnp.int8)   # -1
```

`intentional_wrap(value, dtype)` returns the wrapped integer, so the value that
reaches jax is the value jax would have produced anyway — the program is
byte-identical with the detector on, off, or uninstalled. It needs no jax and
no numpy, and it is recorded: the session report prints every declaration with
its site.

Three properties, and they are why this shape and not a flag or a suppression
list:

* **it is exact**, because you assert it rather than the tool inferring it;
* **it cannot license a different site** — it is a value, not a mode;
* **it cannot hide a truncation at a different dtype**. The dtype is half the
  declaration, and sometimes a declaration that drifted from its use fires:
  `intentional_wrap(0xFF, "int8")` is `-1`, which `uint8` cannot hold. Often
  it does not: `intentional_wrap(300, "int8")` is `44`, which every other
  integer dtype holds — 53 of 98 measured (declaration, misuse) pairs pass
  silently. What is guaranteed is narrower and is the part that matters: in
  every silent case the declared value is *in range* at the new dtype, so
  nothing is narrowed there and no truncation is hidden. Writing the wrong
  constant is a bug; it is not one this detector claims to catch.

**There is no value-based exemption, and there will not be one.**
`jnp.full((4,), 0xFF, jnp.int8)` and `jnp.full((4,), 255, jnp.int8)` produce
*identical* observations at the hook — same written value, same dtype, same
result, same frame — and differ only in source text, which a variable, an
imported constant or a computed value does not carry at all, so a rule that
read the line would work for literals and abstain for exactly the mask idiom
it was meant to recognise. Two candidate heuristics were driven over a
corpus of real narrowings: "a negative into an unsigned dtype is deliberate"
hard-errors correct code 7 times and lets a real bug through once; "an all-ones
result is deliberate" is 5 and 2. Both are wrong in both directions.

For code whose *subject* is the truncation — a test that demonstrates a door
narrows in silence, a reproducer in a disclosure — there is
`stelling._tripwire.eager.expected_truncation(reason)`, a context manager
taking a mandatory reason. Everything it permits is counted and printed with
that reason. It is deliberately the awkward one.

It is **dynamically scoped**, not lexically bounded: the region is open from
`__enter__` to `__exit__`, in whatever code runs between them. Another thread
is not licensed, and neither is another `asyncio` task on the same loop — the
region stack is a `contextvars.ContextVar`. A **generator** suspended inside a
region is the exception and it is not fixable: a plain generator shares its
caller's context, so a region it entered and has not left is open in the code
that resumed it. Prefer `intentional_wrap`, which has no scope beyond the
expression it is written in.

### What it closes, and what it does not

Seven of the **nine** `unwatched` routes in
`tests/test_tripwire_gate_coverage.py::GATE_COVERAGE` move from "silently
certifies a destroyed constant" to "cannot be traced at all": `jnp.full`,
`jnp.full_like`, `lax.full`, `lax.full_like`, `lax.convert_element_type`,
`jnp.stack`-of-`full` and `lax.select`-of-`full`.

*This page said "six of the seven" until 2026-08-20, in a paragraph whose
next line says two routes remain — 6 + 2 is 8, and the denominator was
simply never re-read after `fc98241` made `jnp.stack`-of-`full` the eighth
`unwatched` row. Both halves are read now, by
`test_the_documented_fraction_is_the_measured_one`, in this page and in
the five other files that state the same fraction — asserting the
denominator in Python changed nothing, because nothing read the sentence.
Measured: `GATE_COVERAGE` carries 35 routes, 17 `watched`, 9 `unwatched`,
3 `loud`, 6 `deferred`.*

*The denominator moved 8 → 9 on 2026-08-21 because `lax.select`-of-`full`
became a row. It had been driven closed and disclosed as closed while
being a row of NEITHER inventory, so a jax release that changed it would
have reddened nothing.*

**`jnp.take`'s `fill_value` was disclosed beside it and is a different
case, measured.** Under a TRACE the written constant reaches the jaxpr
intact — driven in three spellings, all `deferred` — so it was never one
of the gate's holes and is not in the fraction above; the propagation's
`convert_element_type` transfer is what declines it. Run EAGERLY there is
no trace to reach it in, the fill array is built at the construction site,
and the detector raises. It is `deferred` in `GATE_COVERAGE` and `raises`
in `EAGER_COVERAGE`, the only row that is both. A scoped
`with jax.disable_jit():` closes too, and for the same reason as `full`:
the constant it wraps is narrowed at this very site on the way in.

Two named routes remain, and both are numpy finishing before jax is reached:

* `np.asarray(N).astype(dt)` is **permanently** unhookable —
  `np.ndarray.astype` is an immutable type attribute, so there is nothing to
  patch, and numpy emits no warning for it even under
  `warnings.simplefilter("error")`;
* `jnp.asarray(np.array(N), dtype=dt)` is a second spelling into the same
  residue.

Both are declared `unwatched` in `GATE_COVERAGE`, declared `silent` in
`EAGER_COVERAGE` beside it, and a test asserts the residue is exactly those
two — so a third cannot join them quietly.

**And most of the table above is untouched by this**, which is worth saying
plainly because "six of eight" invites the wrong reading. The rows this
detector does NOT close, each re-measured with it armed:

* **eager execution of the inline door** — `a + 256` outside `jit` still
  wraps, and the construction site is reached **0** times. The constant is
  promoted to an array before it gets there;
* **`jnp.where(pred, N, x)`, `jnp.clip` at either bound, `jnp.pad`** — 0
  conversions at this site, for the same reason they reach 0 invocations of
  the const-fold rule: the literal sits at the enclosing call site and the
  narrowing happens on a sub-jaxpr *variable*;
* **`x % N`, `x // N`, `jnp.searchsorted`** — same mechanism, same answer.

Those are `report.UNCOVERED`'s bullets 1, 2 and 3, and they stay open. What
closes is bullet 4 except its numpy clause, and bullet 5's scoped
`disable_jit` door.

**With `jit` off, jax narrows its OWN constants eagerly**, and this detector
sits where that happens. Its threefry PRNG mask is `4294967295 -> -1` at
`int32`, and a rule that raised on it would stop every test that touches
`jax.random` under `jax.disable_jit()` or `JAX_DISABLE_JIT=1` — and, because
`chex.fake_jit()` installs `jax.disable_jit()` around a test body, **also with
jax's own defaults untouched**: any suite using `chex.variants(without_jit=True)`
in its default configuration meets jax's eager mask with `jit` ON. That is not
a `JAX_DISABLE_JIT=1` special case; it is a public chex API, confirmed at
`chex/_src/fake.py:256`. Such a rule would ask you to declare a constant you
never wrote, sometimes at a line inside a library you cannot edit.

So the detector does not GUESS whether a constant is jax's. **It looks it up.**
A map records what jax writes, keyed on the jax function that writes it, the
exact value, the dtype the value ARRIVES in, and the dtype it is narrowed to:

```
("_src/random/threefry2x32.py", "_threefry_seed"): 4294967295, uint32 -> int32
```

A narrowing is attributed to jax when, and only when, one of those functions is
in the unbroken run of jax frames beneath your line AND the value and both
dtypes are that row's. **Everything else is yours and raises.** Each
suppression is counted and printed with the site of your call, the jax function
that wrote the constant, and what the constant is.

**The source dtype is in the key because without it a row can suppress YOUR
constant.** At that one site your seed and jax's mask collide:
`jax.extend.random.threefry_prng_impl.seed(np.int64(2**32 - 1))`, **under
`jax.disable_jit()`**, narrows twice under `_threefry_seed` — your seed and
jax's mask, both `4294967295 -> -1` at `int32` — and a row without the source
dtype suppressed both, then printed
*"written by jax … the threefry PRNG's 32-bit mask"* at your own line. They
differ in one field the hook can see: all 13 of jax's own truncations arrive
from `uint32`, and a seed you pass arrives from `int64`.

That map has **one row**, and that is the measurement rather than an omission.
A sweep of jax's own integer surface under `disable_jit` — every key
implementation and seed spelling, then `jax.random`'s consumers and `jnp`'s
integer ops over six integer dtypes — sees 675 conversions and 13 truncations
of jax's own, and **all 13 are that one row**, identically on jax 0.11.0 and
0.10.2. It is shipped code, not a note: it runs as a test on both jax series,
so a release that adds a second one turns a lane red, and **both legs of the
nightly jax canary run it too, at `JAX_ENABLE_X64=0`** — which is the only
setting it can find anything at. With x64 on, jax's mask widens to `int64`, it
fits, and nothing of jax's narrows: the sweep sees 729 conversions, 0
truncations and 0 rows exercised, against 675 / 13 / 1 with x64 off (jax
0.11.0), so its zeroes there mean it did not look rather than that it looked
and found nothing. For one commit both canary legs ran at x64 on only, which
made that page an alarm wired to a condition that could not occur; each leg
now runs the canary in both cells and the x64-on run prints the
qualification.

**The residue is the one an enumeration has, and it is loud.** A jax release
that adds an internal eager truncation nobody has written a row for is
attributed to *whoever called jax* and raises, at a line inside jax you did not
write. The alarm says so in its own message and prints jax's own frames beneath
your line, so a report is one paste away. That is the direction this instrument
fails in on purpose: an over-report is visible to you, holding the quoted line;
a suppression is not.

**The other residue is a collision, and it is the quiet one.** A row is a value
lookup, not a proof of authorship, so a narrowing of yours that agrees with a
row in every field — value, source dtype, target dtype, and a jax function in
the run — is attributed to jax and does not raise. No route this repository has
measured reaches that state any more (a `uint32` seed of the same value
promotes to `uint32` and does not narrow at all), but a sweep is a sample and
the shape stands. It is `report.EAGER_UNCOVERED`'s bullet on collisions.

### A seed that does not survive, which this detector cannot see

**`jax.random.PRNGKey(2**32)` is `jax.random.PRNGKey(0)`, silently, and nothing
here says so.** `PRNGKey` and `key` cast the seed with
`jnp.asarray(np.int64(seeds))` inside jax's own `random_seed`
(`jax/_src/random/prng.py`, line 558 on jax 0.11.0), and that cast is
NUMPY-level — it happens before the construction site this detector patches, so
the hook records **zero** observations of your seed, with `jit` on and with
`jit` off alike. Measured on jax 0.11.0 with x64 off:

```
PRNGKey(2**32 - 1) == PRNGKey(-1)     True
PRNGKey(2**32)     == PRNGKey(0)      True
PRNGKey(2**33 + 5) == PRNGKey(5)      True
```

The one observation that program *does* produce at the hook is jax's own mask
(`4294967295` from `uint32`, `jit` off only), which is correctly suppressed. So
**`no alarm` on `jax.random.PRNGKey(N)` means "stelling saw nothing of yours",
not "your seed survived."** A seed wider than `int32` is not covered, and
closing it needs a hook at a numpy cast rather than at jax's constructor —
a different design, not a patch. It is disclosed in the section's own coverage
list.

**The lower-level entry point is the same story with `jit` ON — and it is the
program the source-dtype fix above is sold on.** `_threefry_seed` is `@jit`, so
in the **default** configuration your seed is canonicalised to `int32` at jax's
argument boundary, before any trace begins. Measured on jax 0.11.0, x64 off:

```
threefry_prng_impl.seed(np.int64(2**32 - 1))   ==  seed(np.int64(-1))   True
                                                   both [0 4294967295]

  jit ON   eager detector:      0 conversions of your seed, no alarm
           const-fold tripwire: 1 narrowing — jax's own mask at
                                threefry2x32.py:73, uint32 4294967295,
                                correctly suppressed
  jit OFF  eager detector:      1 conversion, 1 truncation, RAISES at your
                                line, 4294967295 -> -1
```

So **with jax's defaults that call kills your seed and neither instrument
reports it.** Turning `jit` off is what makes it visible, which means the
collision worked through above is a `disable_jit` phenomenon; with
`JAX_ENABLE_X64=1` the seed survives and there is nothing to report. Closing
this is the same numpy-cast hook as the paragraph above, and it is not
attempted here.

### It fails closed

This patches a private jax function, so drift must refuse rather than stop
watching. Arming verifies the module and the attribute are where they are
expected, that `inspect.signature`'s first two parameters are still
`operand` and `new_dtype` and still passable positionally, and then **drives
every construction route it claims, in both directions**. If one route stops
reaching the site — the silent failure a jax release actually produces — it
reports `route-blind:<route>` and does not attach. Keeping five routes and
losing one quietly is not a trade the tool makes on your behalf.

Arming also drives **the attribution itself**, which the route probes cannot:
they swap in a collector, so they never reach the function that decides whether
a narrowing raises. One narrowing of each origin goes through the live policy —
a constant written at no enumerated jax site, which must raise, and
`jax.random.key(0)` under `jax.disable_jit()`, which must be attributed to jax.
If the second one stops holding, the map no longer names jax's own mask, the
next `jax.random` call under `disable_jit` would raise inside jax, and arming
reports `origin-blind:jax-attributed-to-you` and does not attach. The control
leaves your counters exactly as it found them.

The nightly jax canary arms it, drives a live control both ways, and checks
the site's source hash against a per-release map (`_KNOWN_EAGER_HASHES`, keyed
on the exact release, recorded and never gated on). It also asks, once and
while both hooks are live, the one displacement question that covers **both**
of them: is stelling's wrapper still the live one? A `no` there is
`hooks:displaced` and it names which.

## Reading the report

**The denominator is always printed**, in the form the tool actually prints it:

```
denominator: 3011 integer const-folds inspected (3288 constants folded, 9412 rule invocations).
```

A count is evidence; `0 findings` on its own is indistinguishable from a dead
hook, and this tool will not print the second one either. A run whose
denominator is zero says so in as many words.

**Suppressed narrowings are named, not dropped.** JAX's own PRNG seed mask
(`4294967295 -> -1`, `int32`, at `JAX_ENABLE_X64=0`) is a real narrowing
written by jax inside jax, and blaming your `jax.random.key(...)` line for it
would be wrong. It is filtered out of the findings and listed separately, with
the jax file and line that wrote it, so that a filter and a blind instrument
do not look the same.

**"Outside jax" is not the same as "yours".** The filter has exactly one
boundary — jax's own source tree — so a constant written inside a third-party
library you called is a finding, reported at *that library's* file and line.
Driven with a real module in a venv's `site-packages`: the site is named
correctly and is checkable, which is why the report quotes the writing line and
prints the call chain instead of telling you what you wrote.

**Findings fire once per trace, not once per call.** Twenty calls of one
jitted function is one finding. Two runs of one suite produce byte-identical
reports: findings are sorted by `(file, line, value)`, never by the order they
fired.

**Nothing is ever auto-fixed.** The right fix depends on intent that is not in
your source: `q - 128` on `uint8` in a quantization routine is a real finding
where "widen the dtype" and "this saturation is deliberate" are both
defensible. The report offers both branches and labels them as inference.

## If it cannot arm

The tripwire attaches to a **private** JAX registry. No public or
`jax.extend` surface exports it — seven candidate modules were measured on
both tested series and none does — so a JAX release can move it out from under
this tool at any time. When that happens:

* your suite **still passes**, with the exit code unchanged;
* the summary says `NOT ARMED [<code>]` and what the code means;
* it says what still works, because "disabled" must never read as "you are
  unprotected". Static checking is unaffected: `stelling.preconditions.check`
  and every verdict path work exactly as before.

The codes are stable and greppable: `no-module`, `no-registry`, `no-entry`,
`not-invoked`, `cries-wolf`, `mis-attributed`, `below-floor`, `foreign-patch`,
`detached`, `no-worker-reported`, `mixed`, `no-site-module`, `no-site`,
`signature-drift`, `route-blind`, `unexpected:<ExcType>`.

`no-worker-reported` and `mixed` belong to an xdist **controller**, which never
arms and whose status is its workers' agreement: the first when not one worker
sent a status back, the second when they disagreed.

The last four belong to the **eager construction-site detector**, which
attaches to a private jax *function* rather than a registry entry:
`no-site-module` and `no-site` are the module and the attribute not being where
they are expected; `signature-drift` is the function being there with its first
two parameters moved, which a presence check would pass; and
`route-blind:<route>` is the one that matters — the function is there, the hook
is installed, and jax has stopped sending one construction route through it.
That last failure is silent, it is what a jax release actually produces, and
the detector refuses to attach rather than watching five routes and quietly
losing the sixth.

### It can also stop being armed part-way through

The status is checked again at the END of the session, not only when it arms,
because a hook that left the registry in the middle of a run is the case where
`armed` over a small denominator is most misleading. `detached` means it was
taken back out — a mid-run `disarm()`, or a nested pytest session that enabled
the tripwire and restored the original when it finished — and `foreign-patch`
means something rebound the registry over the top of it.

Either way the figures already collected are still printed, under a **PARTIAL**
banner in the same words a lost xdist worker gets: they cover the instrumented
part of the run and no more. Under `--stelling-overflow=require` the session
fails, because it did not stay armed.

The message never travels by `warnings.warn`. Under `-W error::UserWarning` —
common in scientific repos — a "safely disabled" warning becomes an exception
and crashes the suite this guardrail exists to protect. It goes through
pytest's terminal reporter, which cannot be escalated.

**If you depend on it**, take the escalation yourself:

```console
$ pytest --stelling-overflow=require
```

which fails the session if the tripwire cannot arm. That decision is yours and
is deliberately not the default.

## Under `pytest-xdist`

Aggregation works: workers serialise their findings back and the controller
reports the true total, deduplicated across workers. The controller itself is
deliberately **not** instrumented — under `-n auto` it runs no tests, so
arming there would import JAX into a process that never traces.

Workers **expected** are counted against workers **reported**. If they differ
— a worker crashed and never sent its findings — the report says
`LOST WORKERS: 3 of 4 workers reported ... a PARTIAL, not a total` rather than
presenting a partial sum as a total.

## What it costs

Nothing until you switch it on. The plugin is registered by every `pip install
stelling`, and until one of the three opt-ins above says otherwise it adds one
command-line flag, imports no JAX, and installs no hook. `import stelling`
still pulls in no JAX at all.

Once on: the hook is a thin wrapper over a constant-folding rule, and it walks
a stack only when a narrowing is actually out of range. It never raises into
your trace — if its own bookkeeping fails it counts the failure, discloses it
in the report, and carries on, because an instrument that breaks the suite it
is measuring is worse than no instrument.

**One real side effect, and it is on your jit caches.** While the tripwire is
armed, every `preconditions.check()` calls `jax.clear_caches()` before the
trace it gates, and so does every `contracts.check_contract()`, which reaches
the same gate. That call is what makes the observation complete rather than
partial, against jax's own caches and on one thread (see the warm-trace-cache
row above and the two rows under it). `clear_caches` is process-global,
so it also drops *your* compiled functions, and the next call to one of them
pays its trace and compile again. **The call itself scales with how many
jitted functions are live**, so a single "populated" figure means nothing
without the size; measured on jax 0.11.0 (median of 12, four-primitive
functions):

| live jitted functions | `jax.clear_caches()` |
|---|---|
| 0 (empty) | 0.049 ms |
| 1 | 1.4 ms |
| 10 | 8.3 ms |
| 50 | **41.8 ms** (39.6–48.3) |

The re-compile a caller then pays is 18 ms for a trivial jitted function,
45 ms for a 32-step `scan`, and 330 ms for a 200-primitive chain. On this
repository's own suite and `corpus/` — 1475 gated traces — it was not
measurable above the noise (522.0 s → 519.7 s), because this suite holds few
jitted functions live across a `check()`; a caller that holds fifty will not
see that. If your suite interleaves `check()` with an expensive
jitted model, budget one full re-compile of that model per `check()`, and note
that none of this happens unless the tripwire is armed.

## What it does to your verdicts

The gate has **three** states, and the third one is not a shade of the first.

**Observed, clean.** Nothing narrowed and the whole trace was watched, so the
pipeline proceeds and the verdict is whatever the analysis finds.

**Observed, narrowed.** A narrowing fired during a `stelling.harness.trace()`
call, so the verdict is **UNKNOWN** — the pipeline refuses to propagate or
judge a jaxpr that does not represent the program as written. The note says how
many narrowings were detected and directs you to the tripwire report for
details:

```
trace unfaithful: 1 integer narrowing(s) detected during tracing — ...
```

**Not fully observed.** The tripwire could not watch all of the trace, so the
run has no evidence *either way* about the part it did not watch. The verdict
is **UNKNOWN** and says so in its own words, because "no narrowing was seen"
and "no narrowing occurred" are different claims and sending you to hunt a
narrowed constant that nobody looked for wastes your afternoon:

```
trace NOT FULLY OBSERVED: the overflow tripwire could not watch all of this
trace, so this run has no evidence either way about the part it did not watch.
THIS IS NOT A REPORT THAT A CONSTANT WAS NARROWED — ...
```

It is reached when jax's trace caches could not be emptied, or when the
tripwire was disarmed or re-armed while a harness was being traced. When both
the second and the third state apply at once, the count is published as a
**lower bound**: fixing the narrowing it names is not evidence that it was the
only one.

When the tripwire is NOT armed (the default), this gate is inactive and
verdicts are unaffected. The gate is a function of the tripwire's state at
trace time, never of whether the plugin is installed.

## Configuration reference

| | |
|---|---|
| `pytest_plugins = ["stelling.overflow"]` in `conftest.py` | switch on, mode `auto` |
| `-p stelling.overflow` | switch on, mode `auto` |
| `--stelling-overflow=auto` | switch on; report if it cannot arm, exit code unaffected |
| `--stelling-overflow=require` | switch on; **fail the session** if it cannot arm |
| `--stelling-overflow=off` | off, even if the plugin above is loaded |
| `--stelling-eager-truncation=error` | switch on the **eager construction-site detector**; an undeclared truncation raises, and the session **fails** if the detector cannot attach |
| `--stelling-eager-truncation=off` | off — the default, and `--stelling-overflow` does not change it |
| *(nothing)* | both off — the default |

### If your CI sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`

`--stelling-overflow` is a flag **registered by a plugin**, and the plugin
reaches your session through a `pytest11` entry point. That entry point is what
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — a common CI hygiene setting — switches off.

The first two spellings above work anyway: naming `stelling.overflow`, in
`conftest.py` or with `-p`, loads the plugin itself. **The flag on its own does
not**, and it does not fail quietly — pytest exits 4 with
`unrecognized arguments: --stelling-overflow`. Name the module as well:

```console
$ pytest -p stelling.overflow --stelling-overflow=require
```

**`--stelling-eager-truncation` is registered by the same plugin and has the
same property.** Naming `stelling.overflow` loads the plugin without arming
the tripwire — the two dials are independent — so this is the spelling for the
eager detector alone under `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`:

```console
$ pytest -p stelling.overflow --stelling-overflow=off --stelling-eager-truncation=error
```

Tested against JAX 0.10.2 and 0.11.0. A newer JAX arms anyway and the report
says so; the probe is the contract and the version is a disclosure.
