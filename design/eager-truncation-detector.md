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

**IT NEEDS NO ORIGIN FILTER.** The const-fold tripwire fires on jax's OWN
constants — `jax.random.key(0)` folds `4294967295 -> -1` inside
`threefry2x32.py` — and carries `record.attribute`, an `ORIGIN_JAX` bucket and
a suppressed-findings report to keep that out of a user's results. The eager
detector never sees it: measured, the same program gives 2 scalar conversions
and 0 truncations. That is a property of where the two hooks sit — jax's
internals build their masks as arrays or as in-range scalars, not as
out-of-range Python integers handed to a constructor — and it is what makes an
alarm that STOPS the program affordable at all. An instrument that raised
inside jax's own PRNG would be unusable at any blast radius.

**The cost, stated rather than discovered.** `finally:` blocks still run and
context managers still exit, so ordinary cleanup is unaffected. Cleanup written
as `except Exception: release()` does NOT run, so a caller who releases a
resource there and not in a `finally:` will leak it. pytest reports a
`BaseException` from a test body as an ERROR rather than a FAILURE, which is
the right shape: a truncated constant is not a failed assertion, it is a
program that cannot be measured.

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
dtype, the same result, the same frame. They differ only in source TEXT, which
is not available at the point the decision has to be made, and which a
variable, a computed value or a constant defined in another module does not
carry at all. Intent is therefore not a function of `(value, dtype, result)`,
and no rule over that data can be sound.

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
* **cannot license a different dtype**: the dtype is half the declaration.
  `intentional_wrap(0xFF, "int8")` is `-1`, out of range for `uint8`, so a
  declaration that drifted from its use fires rather than passing.

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

It is deliberately the awkward one: mandatory reason, lexically bounded,
thread-local, and **every truncation it permits is counted and printed with its
site and its reason**. An opt-out that hid what it suppressed would reintroduce
the same silence one level up. It lives in `stelling._tripwire.eager` and not
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
`report.UNCOVERED`'s bullets 1, 2 and 3 and they stay open.

## The numpy fence, abandoned by ruling

`DynamicJaxprTrace.new_const` was measured as a genuine single choke point for
numpy ingestion. It is not usable: a plain Python `x + 0` arrives there as a
numpy `int8` array indistinguishable from a user's, so a narrow-integer fence
would reject every integer literal in every jax program. Recorded here so the
next person does not re-derive it.
