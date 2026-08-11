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

The `256` you wrote is destroyed. No error, no warning, no `RuntimeWarning`
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
| **`lax.convert_element_type(N, dt)`** | **UNCOVERED** | as above |
| **`lax.select(p, jnp.full(shape, N, dt), x)`** | **UNCOVERED** | as above |
| an operand that was already an array | **UNCOVERED** | the fold declines non-scalars, so the wrap already happened |
| **inside `with jax.disable_jit():`** | **UNCOVERED** | 0 fires on a jaxpr *byte-identical* to one that fires outside the block |
| anything traced **before** the plugin armed | **UNCOVERED** | jit caches; it is never re-traced |

There are two distinct causes, and the second is the one worth knowing about.

**The site is never reached.** `where`, `clip` at either bound, `pad` and
`take` never fold a constant here: the literal sits at the enclosing call site
and the `convert_element_type` inside the sub-jaxpr operates on a **variable**.
Other mechanisms may reach these; none is built.

**The value is already narrowed before the site.** `jnp.full`, `jnp.full_like`,
`lax.convert_element_type`, `lax.select` and a scoped `jax.disable_jit()` all
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
`detached`, `no-worker-reported`, `mixed`, `unexpected:<ExcType>`.

The last two belong to an xdist **controller**, which never arms and whose
status is its workers' agreement: `no-worker-reported` when not one worker
sent a status back, `mixed` when they disagreed.

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

## What it does not do to your verdicts

Nothing. A fire does **not** change any stelling verdict today. If the
tripwire fires while stelling is tracing a harness, the trace is not faithful
to the source as written and the query arguably should be downgraded to
UNKNOWN — but that is a change to what VERIFIED means, it is reserved, and it
is not wired. Do not read a green stelling verdict as "and no narrowing
occurred".

## Configuration reference

| | |
|---|---|
| `pytest_plugins = ["stelling.overflow"]` in `conftest.py` | switch on, mode `auto` |
| `-p stelling.overflow` | switch on, mode `auto` |
| `--stelling-overflow=auto` | switch on; report if it cannot arm, exit code unaffected |
| `--stelling-overflow=require` | switch on; **fail the session** if it cannot arm |
| `--stelling-overflow=off` | off, even if the plugin above is loaded |
| *(nothing)* | off — the default |

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

Tested against JAX 0.10.2 and 0.11.0. A newer JAX arms anyway and the report
says so; the probe is the contract and the version is a disclosure.
