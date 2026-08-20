# D4 — The wrap disclosure and launch narrative

**Status:** Pre-release reference. Grounded in measurements from the prior-art
sweep (2026-08-11) and the adversarial audit sequence that hardened the gate.
Every claim below is cited; the appendix has the URLs.

---

## The core finding

JAX silently wraps out-of-dtype-range Python integer constants on their way into
a trace. On an `int8` array:

```python
y = x + 256   # reaches the jaxpr as: add a 0:i8[]
```

The `256` you wrote becomes `0`. No error, no warning, no mechanism you can
enable to make it loud. This is not a bug — it is a **deliberate architectural
decision**, pinned in JAX commit [`c2fe350455`](https://github.com/jax-ml/jax/commit/c2fe350455)
(2023-04-04, *"future-proof lax.convert_element_type"*), which created the
wrapping `.astype` and added a test asserting the wrapped value in the same diff.
It is the only test in JAX's 23,705-case suite that asserts a wrapped value;
every other overflow test asserts a raise.

The reason it exists: JAX's own PRNG mask (`4294967295 → -1`, `int32` at
`JAX_ENABLE_X64=0`) passes through this site on every `jax.random.key()` call.
It is the one constant that needs the wrap, and the 24 others measured in the
13-spelling × 3-door grid are collateral.

## Why this matters for verification

Any tool that verifies properties of JAX programs from the **jaxpr** — the
traced intermediate representation — operates on a trace where the literal has
already been destroyed. A verifier can return **VERIFIED for a predicate that is
false of the program as written**. The written constant and the traced constant
are different numbers, and nothing in the ecosystem reports the disagreement.

This is not a hypothetical: stelling is such a verifier, and this is a measured
gap in its own soundness — the reason the tripwire was built.

## What we tried (the landscape, measured)

Six mechanisms that look like they should help, and don't:

| mechanism | result |
|---|---|
| `jax.numpy_dtype_promotion("strict")` | separates **dtypes**, not values; raises on in-range `np.int64(3)`, silent on out-of-range `256` |
| `jax.config.update("jax_enable_checks", True)` | silent |
| `debug_nans` / `debug_infs` | silent |
| `np.errstate(over='raise')` | silent |
| `warnings.simplefilter('error')` | silent |
| `jax.experimental.checkify` with `all_checks` | returns `None` while OOB and div-by-zero controls throw |

(All measured on JAX 0.11.0 and 0.10.2, x64 on and off.)

**checkify** deserves special mention because it is the closest thing JAX has to
a safety layer. It is **structurally blind** to this class: constant folding
erases the literal *within* the trace before checkify's transform runs over the
jaxpr. The value is already gone by the time checkify sees anything. Confirmed on
JAX 0.11.0:

```
>>> checked_f = checkify.checkify(f, errors=checkify.all_checks)
>>> err, result = jax.jit(checked_f)(jnp.int8(5))
>>> print(err.get())          # None — checkify sees nothing
>>> print(result)             # 5 (should be 261 mod 256 = 5... but 256 mod 256 = 0)
```

**`jax_check_static_indices`** reaches static constants only — gather operations
with literal indices. It does not cover arithmetic narrowing.

**`error_checking_behavior`** (`jax._src.numpy.error`, no public API in 0.11.0)
catches division by zero, NaN generation, and array-valued gather OOB — but NOT
integer overflow wrapping, NOT scalar-index OOB, and NOT `dynamic_slice` OOB.
Confirmed on 0.11.0: `error_checking_behavior(all='raise')` leaves `x + 200` on
int8 silent (result: 44, no error) while correctly catching `1 // 0`. It has no
public surface today; it may gain one in a future JAX release, but its scope
(div/nan/oob) does not include overflow.

## JAX's own history with this problem

The JAX team has engaged with this problem and found it intractable from their
side:

- [**PR #15275**](https://github.com/jax-ml/jax/pull/15275) (2023-03-28, mattjj):
  *"add overflow errors for numpy.ndarray"*. Closed without merge after 8
  months — test failures from silently-tolerated overflows in existing JAX code.

- [**Issue #31426**](https://github.com/jax-ml/jax/issues/31426) (2025-08-28):
  *"Edit lax.convert_element_type to emit deprecation warning on overflow"*.
  Still open, zero comments.

- [**PR #34797**](https://github.com/jax-ml/jax/pull/34797) (2026-02-03,
  community): Attempted to fix #31426 by emitting a `DeprecationWarning` in
  `convert_element_type`. **Closed within 44 minutes** by jakevdp:
  *"fundamentally incorrect approach... operand may be traced and so np.asarray
  will raise ConcretizationTypeError."*

- **Justin Fu** (Google, 2026-02-12,
  [#35013 comment](https://github.com/jax-ml/jax/issues/35013#issuecomment-3892622411)):
  *"I would rather there be an error for OOB indexing if it's statically
  provable instead of silently giving the wrong answer."*

The rejection of PR #34797 is the key insight: a naive warning at
`convert_element_type` is "fundamentally incorrect" because the operand may be a
traced value. For a constant that reaches the trace, the detection has to happen
at the **constant-folding rule**, which is the one site where the value is known
to be concrete and is about to be narrowed. That is where the tripwire sits.

*"The detection has to happen at the constant-folding rule" stood unqualified
until 2026-08-20 and was too wide by one door.* A constant that never reaches
the trace has already died at ARRAY CONSTRUCTION, and there is a second
concrete-by-construction site there —
`design/eager-truncation-detector.md`'s, opt-in and off by default. The
argument above is the argument for the INLINE door, and it is sound for it.

## What stelling ships (0.1.0)

### The tripwire: `pytest -p stelling.overflow`

One line in `conftest.py`, or one CLI flag. Your existing test suite, unchanged.
The tripwire hooks `const_fold_rules[convert_element_type_p]` — the exact site
where the value dies — and reports each narrowing with:

- the source location (innermost frame outside JAX)
- the written value and the narrowed value
- the arithmetic (`300 mod 2^8 = 44`)
- an independent recomputation confirming the finding
- a one-line reproducer (`jax.make_jaxpr(...)`)

Every finding is **recomputed by an independent route before it is printed**. If
the hook's observation and the recomputation disagree, the disagreement is
printed and the finding is withheld.

**What it costs:** nothing until you switch it on. Once on: within measurement
noise over 60 cold traces (2.66 s vs 2.57 s). It never raises into your trace —
if its own bookkeeping fails, it counts the failure, discloses it, and continues.

### The static verifier

Forward interval propagation over the jax-free IR, outward-rounded, escalating
undecided obligations to an SMT portfolio (cvc5/Z3). Every verdict carries a
full stamp naming its assumptions, precision, and what it dropped.

When the tripwire is armed and a narrowing fires during a harness trace, the
verifier refuses to propagate or judge the jaxpr — the verdict is **UNKNOWN**
with a note naming the narrowings. It refuses separately, and in different
words, when it could not watch the whole trace: "no narrowing was seen" and
"no narrowing occurred" are different claims and only the second licenses a
VERIFIED.

So a VERIFIED verdict, with the tripwire armed, is a statement that the
property holds over the declared set AND that no narrowing was seen on any
route the tripwire watches. The two checks compose: detection gates proof.
**The watched set is finite, and the qualifier is the whole of the honest
claim.** A whole bucket of construction routes is `unwatched` —
`jnp.full(shape, N, dt)`, `jnp.full_like`, `lax.full`, `lax.full_like`,
`lax.convert_element_type(N, dt)`, `np.asarray(N).astype(dt)`, and anything
built on `full` such as `jnp.stack` or `lax.select` — because the value is
narrowed at array CONSTRUCTION and the fold rule is handed something already
in range. Those programs get a VERIFIED about a constant that no longer
exists.

*The mechanism in that sentence read "because numpy narrows the value before
any jax primitive runs" until 2026-08-20, and it is measured wrong for all but
two of the routes it names.* Six of them narrow INSIDE jax, at one line of
`jax._src.lax.lax._convert_element_type` — numpy does the arithmetic, but
inside a jax function, which is why a hook can sit there at all. Only
`np.asarray(N).astype(dt)` and `jnp.asarray(np.array(N), dtype=dt)` finish
before jax is reached, and those two are the permanent residue. The
correction matters because the false mechanism was the reason this page gave
for the door being unhookable: `design/eager-truncation-detector.md` hooks
it, opt-in and off by default, and this paragraph's claim about the DEFAULT
configuration is unchanged. The full inventory, one bucket per route and measured rather
than typed, is `tests/test_tripwire_gate_coverage.py::GATE_COVERAGE`; the
user-facing version is `docs/overflow-tripwire.md`.

The gate is hardened against the edge cases adversarial audit surfaced:
- **jax.make_jaxpr's identity cache** (a repeated trace returns the cached
  jaxpr without re-entering the fold rule) is defeated by wrapping each trace
  in a fresh closure
- **Nested `check()` calls** (a harness that internally verifies sub-properties)
  are isolated by a thread-local LIFO stack of per-invocation fire counters —
  each gate counts only fires from its own trace, never from an inner or outer
- **Disarm/rearm during trace** is detected by recorder-identity comparison —
  if the recorder object changes, the gate refuses
- **selfcheck's probe** (which fires through the same wrapper) is isolated by
  its own stack entry, so arming inside an active gate does not contaminate it

## What this means in practice (concrete stakes)

The abstract statement — "a verifier can certify a property that is false of the
program" — translates to physical consequences the moment the verified program
controls something real. One worked example, from the class of programs stelling
is built for:

### A pressure-relief controller verified against a corrupted trace

A JAX-traced controller computes a scaled sensor reading for a chemical reactor's
pressure-relief valve. The scaling factor — a unit-conversion constant — is
40,000 (counts-to-millibar on a 16-bit ADC):

```python
def pressure_mb(raw_adc):        # raw_adc: int16[...]
    # Convert 16-bit ADC counts to millibar
    return raw_adc + 40000
```

The constant is written as a bare Python `int` against an `int16` array, which
is the form that wraps *silently*. **`jnp.int16(40000)` is not that form**: jax
validates the literal against the dtype and raises `OverflowError` before
anything is traced, so it is `loud` in
`tests/test_tripwire_gate_coverage.py::GATE_COVERAGE` and there is nothing for
a tripwire to catch. The contrast is jax's own and is worth knowing: the same
value through `jnp.full(shape, 40000, jnp.int16)` neither raises nor fires.

The safety property being verified: **"reported pressure stays below the
relief-valve trigger (200 mbar)"** — a plausibility check ensuring the sensor
path doesn't report values that would suppress a necessary venting action.

**What actually happens:** `40000` wraps to `-25536` on int16. The jaxpr is
`{ lambda ; a:i16[2]. let b:i16[2] = add a -25536:i16[] in (b,) }`. For raw ADC
readings in `[0, 100]`, the traced program produces `[-25536, -25436]` — all
far below 200. (Measured on jax 0.11.0, x64 on, at this branch.)

**Without the gate:** the verifier propagates the jaxpr, finds all values below
200, returns **VERIFIED**. The CI pipeline sees green. The controller deploys.

**In production:** the real hardware adds 40,000, producing readings in
`[40000, 40100]` — two hundred times the relief-valve trigger. The valve never
opens. The reactor overpressures.

**With the gate:** the tripwire fires during the trace (`40000 → -25536, int16`),
the gate refuses to propagate, and the verdict is **UNKNOWN** with:

```
trace unfaithful: 1 integer narrowing(s) detected during tracing — the jaxpr
does not represent the program as written.
```

The CI pipeline sees amber. The engineer sees the finding, fixes the dtype or
the constant, and the next run either verifies the real property or refutes it
honestly.

### The pattern

The failure mode is not "the verifier has a bug." The failure mode is that the
verifier is **correct about a different program** — one where 40,000 is -25,536,
where 256 is 0, where 300 is 44. Every arithmetic step after the narrowing is
faithfully propagated; the intervals are sound; the SMT encoding is exact. The
only thing wrong is the first number, and everything downstream inherits the lie.

This is why the gate is a **soundness mechanism**, not a convenience feature. A
VERIFIED that attests to a trace that doesn't represent the source is worse than
no verification at all — it actively suppresses the signal that something is
wrong.

## What it does NOT see (the honest table)

Printed on every run, findings or not. This is a floor, not a census.

| door | status | why |
|---|---|---|
| `x + N`, `x * N`, `x >= N`, `x.at[i].set(N)`, `jnp.maximum`, `jnp.minimum` under a trace | **covered** | the fold rule is reached |
| `jnp.where(pred, N, x)` | **UNCOVERED** | the literal sits at the enclosing call site; the fold operates on a variable inside the sub-jaxpr |
| `jnp.clip(x, lo, N)` | **UNCOVERED** | same — nested sub-jaxpr |
| `jnp.full(shape, N, dt)` / `jnp.full_like` | **UNCOVERED by this hook** | the value is narrowed at array construction, inside jax, before the fold site is reached. Closed by the opt-in eager detector, which is off by default |
| eager execution (outside `jit`) | **UNCOVERED** | warm dispatch is 11 frames of C++ fast path; the fold is reached 0 times, and the eager detector sees 0 conversions here too |
| anything traced before the plugin armed | **UNCOVERED here, COVERED inside `check()`** | jit caches, so the body is never re-traced. `preconditions.check()` and `contracts.check_contract()` empty jax's trace caches before the trace they gate, so a *verdict*'s observation is complete with respect to jax's caches on one thread; this *session report* has no such moment. `docs/overflow-tripwire.md` carries the two rows past that qualifier |

*The last two rows' "why" columns are dated 2026-08-20. The third row read a
flat **UNCOVERED** — "jit caches; it is never re-traced" — after the cache
eviction had already closed it for a verdict, so this page and
`docs/overflow-tripwire.md` gave different answers to the same question.*

The two causes are distinct:
1. **The site is never reached** (`where`, `clip`, `pad`): the literal survives
   in the enclosing trace, so standard static analysis is NOT blind to these.
2. **The value is already narrowed** (`full`, `full_like`, eager): the
   narrowing happens at construction — inside jax for the `full` family, in
   numpy before jax for the two `np.asarray` spellings — so the fold rule sees
   an in-range value.

## The prior-art claim (what we are NOT saying)

**We are saying:** no existing tool in the JAX ecosystem performs trace-time
overflow detection for operator-syntax integer narrowings (`x + N`, `x * N`,
`x.at[i].set(N)`, etc. — the spellings where the literal is destroyed before
the trace exists). Confirmed: zero tools on GitHub, PyPI, or in academic
literature. JAX's own attempt was abandoned; the community's was rejected as
fundamentally infeasible.

**We are NOT saying:** static analysis is universally blind to all integer
operations. For `jnp.where` and `jnp.clip`, the literal survives in the
enclosing trace — a future verifier pass could catch those from the jaxpr alone.
The tripwire is necessary for the spellings where the literal is destroyed
*before* the trace exists.

**We are NOT saying:** this replaces checkify. checkify catches OOB indexing and
division by zero at runtime; stelling catches integer-literal narrowing at trace
time. They are complementary, not competing.

## The positioning

Stelling is to JAX what [Kani](https://github.com/model-checking/kani) is to
Rust: assertion-based verification of ordinary programs. Not neural-network
robustness (alpha-beta-CROWN lineage), not runtime instrumentation (checkify),
not type-level checking (jaxtyping). It proves stated properties of traced
computations, and the tripwire is the guardrail on the trace it proves against
— on an enumerated, finite and measured set of routes by which a constant can
be narrowed, not on all of them.

---

## Appendix: citations and source links

| claim | source |
|---|---|
| JAX commit pinning the wrap | [c2fe350455](https://github.com/jax-ml/jax/commit/c2fe350455) (2023-04-04, *"future-proof lax.convert_element_type"*) |
| The pinned test asserting wrapped value | [`tests/lax_test.py:201` `testConvertElementTypeOOB`](https://github.com/jax-ml/jax/commit/c2fe350455) in the same diff |
| 15 raise / 24 wrap grid | measured on 0.11.0 + 0.10.2, x64 on/off; 3 doors × 13 spellings |
| checkify blind, JAX 0.11.0 | confirmed 2026-08-11, `checkify.all_checks` returns None on int8 wrap |
| Justin Fu "statically provable" quote | [jax-ml/jax#35013 (comment)](https://github.com/jax-ml/jax/issues/35013#issuecomment-3892622411) (2026-02-12) |
| PR #15275 — overflow errors, abandoned | [jax-ml/jax#15275](https://github.com/jax-ml/jax/pull/15275) (2023-03, mattjj, closed after 8 months) |
| Issue #31426 — deprecation warning request | [jax-ml/jax#31426](https://github.com/jax-ml/jax/issues/31426) (2025-08, open, zero comments) |
| PR #34797 — community attempt, rejected | [jax-ml/jax#34797](https://github.com/jax-ml/jax/pull/34797) (2026-02-03, closed in 44 min by jakevdp) |
| Issue #278 — OOB indexing (broader context) | [jax-ml/jax#278](https://github.com/jax-ml/jax/issues/278) (2019-01, P2, still open) |
| No tools on PyPI/GitHub/Scholar | prior-art sweep 2026-08-11; zero results for "jaxpr verification", "jax overflow detection", "jax static analysis" |
| `const_fold_rules` not exported publicly | 7 candidate `jax.extend` modules measured on both series, none exports it |
| JAX 0.11.0 is latest release | [PyPI](https://pypi.org/project/jax/) + [GitHub releases](https://github.com/jax-ml/jax/releases), checked 2026-08-11 |
