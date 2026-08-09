# The index-bounds round: `dynamic_slice`, `dynamic_update_slice`, and the gather's dynamic index

*Measured 2026-08-09 on jax 0.11.0 and jax 0.10.2, CPU, `JAX_ENABLE_X64=1`.
Baseline `9564728`.*

## The gap, as it was found

Five harnesses over a 10-element declared array, run on `9564728`:

| harness | result on `9564728` |
|---|---|
| `u[3]` static, in bounds | `discharged` |
| `u[30]` static, **out of bounds** | `unknown`, operand spans `[-inf, inf]` |
| `u[i]` traced, `i ∈ [0,5]`, in bounds | `unknown`, `[-inf, inf]` |
| `u[i]` traced, `i ∈ [15,20]`, **always** out of bounds | `unknown`, `[-inf, inf]` |
| `u[[1]]` array index | `discharged` |

Any dynamic index collapsed to ⊤, and everything downstream of it with it.
**stelling was SAFE throughout** — it withheld rather than modelling jax's
clamp — so this was a power gap, never a soundness gap.

## The measurement that decided the design

**`u[i]` with a traced `i` is not a gather.** `jnp`'s `__getitem__` emits the
from-the-end normalisation and then a `dynamic_slice`:

```
c:bool[] = lt b 0 ; d:i32[] = add b 10 ; e:i32[] = select_n c b d
f:f64[1] = dynamic_slice[slice_sizes=(1,)] a e
```

`dynamic_slice` and `dynamic_update_slice` had **no transfer at all**.
Registering the dynamic-index gather alone would have closed rows 3–5 of the
table above for `u[[i]]` and left `u[i]` — the spelling scientific code
actually uses — exactly where it was.

Two consequences fell out of the same dump:

* **an out-of-range STATIC index takes the dynamic path too.** `u[3]` lowers
  to a static `slice`; `u[30]` and `u[-11]` fall back to
  normalise-then-`dynamic_slice`. So row 2 of the table is closed by the same
  transfer, and the statically-provable out-of-bounds case is reachable.
* **a negative index is Python's from-the-end, at the `jnp` layer.** Measured:
  `arange(10)[-1]` is `9.0`, `[-10]` is `0.0`. The normalisation runs
  *upstream* of the primitive. At the primitive it is the other way round:
  binding `dynamic_slice_p` directly, start `-1` reads element **0** — a
  clamp, not a wrap. Both are true, at different layers, and the transfer
  sits at the lower one, so the interval it classifies is already normalised.

## Three cases

Let the legal start window on axis `d` be `[0, n_d - s_d]` (for a gather row,
`[0, n-1]`).

1. **index range ⊆ the window** → the elementwise hull over every start the
   declared set admits. Tight on a single axis. *The power gain.*
2. **range straddles the window** → **decline**, named. Some declared inputs
   index in bounds and some do not; the out-of-bounds ones take a clamped or
   dropped element, and no box states "this value, or that one, depending".
3. **range disjoint from the window** → **out of bounds for every input the
   user declared**. Reported as a finding, still ⊤.

## Why the clamp is not modelled

Measured, primitive-level: jax **clamps** an out-of-range read
(`dynamic_slice(arange(10), 30, (1,))` → `9.0`) and **drops** an
out-of-range scatter write (`x.at[30].set(v)` on a length-10 `x` is a no-op).

Modelling that would be sound about the **executed** program and wrong about
the program the user **wrote**. The query that separates the two designs:

```python
u = any_array((10,), "float64", (0.0, 1.0))
i = any_array((), "int32", (12, 20))
assert_(u[i] == u[9])
```

Every admitted `i` clamps to 9, so this is **true of what jax runs** — the
test measures that it is — and states nothing about the source. A
clamp-faithful transfer discharges it. This one leaves it undecided.

**The tension is real, not rhetorical.** `SOUNDNESS.md`'s fixed-width
boundary records the tree's posture as *"floats are judged in ℝ; integers and
converts are execution-faithful"*, and jax's clamp is integer index
arithmetic. What decides it the other way is that **there is no single clamp
to be faithful to** — measured, not argued:

1. **One gather, one out-of-range index, two values.** Index 30 into a
   10-element operand: mode `CLIP` returns element 9, mode `FILL_OR_DROP`
   returns the fill value. In range, all three modes agree.

   | mode | `u[30]`, `u = arange(10)` | `u[3]` |
   |---|---|---|
   | `CLIP` | `9.0` | `3.0` |
   | `FILL_OR_DROP` | the fill value (`-1.0` as passed) | `3.0` |
   | `PROMISE_IN_BOUNDS` | `9.0` (UB; this is what CPU happened to do) | `3.0` |

   So "the clamp" is not a property of the operation, it is a property of a
   param — modelling it means picking one of two answers the same jaxpr can
   carry.
2. **Read and write disagree too**: the gather clamps, the scatter DROPS
   (`x.at[30].set(v)` on a length-10 `x` is a no-op, measured), and the same
   source-level `x[i]` picks one or the other by which side of an assignment
   it lands on.

An int32 `add`'s wrap has neither property — it is one defined, reproducible
answer — which is why that is modelled and this is not.

**MEASURED AND NOT A REASON, recorded because an earlier revision of this
page asserted it before running it, and it is false.** Reverse-mode AD *does*
preserve the clamp: the cotangent of `u[30]` on a length-10 `u` lands on
element 9, exactly where the clamped read came from, and the scatter side
agrees (`d/dv` of `.at[30].set(v)` is `0.0`, `d/du` passes all ten through).
The two reasons above stand without it; this one was reasoned and not run,
which is precisely the failure mode the method exists to catch.

The rule therefore computes a value **only where jax's clamp is provably the
identity**. That is the entire soundness argument, and it is why minting a
false VERIFIED through this path requires the *hull* to be wrong rather than
the clamp story.

## The finding channel

Case 3 raises `interval.IndexOutOfBoundsError`, a **subclass** of
`IntervalError`, caught by the walk one arm ahead of the generic decline. The
accounting is deliberately identical — ⊤, `record_unknown`, `mark_unreached`,
never a REFUTED — because an out-of-bounds index does not make any asserted
predicate false, and manufacturing a status from it would claim something the
obligations do not say. Only the note changes, and it is shouted
(`OUT-OF-BOUNDS INDEX (definite)`), the loudest channel a transfer has short
of a new `Stamp` field.

This is the case a jax maintainer asked for in Feb 2026 — *"I would rather
there be an error for OOB indexing if it's statically provable instead of
silently giving the wrong answer"* — and nothing in the ecosystem covers it:
`checkify` is runtime-only, `jax_check_static_indices` reaches static
constants only.

## What was measured

* **Soundness sweep.** Enumerate the WHOLE product of declared start ranges
  over randomised shapes, slice sizes and ranges; execute the real primitive
  at every one; check containment element by element. **6000 configurations
  and ~57 000 executed elements per run, three seeds, both jax series: 0
  containment violations, 0 non-tight configurations.**
* **Positive controls.** Five wrong hulls driven through the same instrument:
  lowest-start-only (908 violations), exclusive upper endpoint (573), axis-0
  only (439), a write rule that never keeps the operand (608), a gather
  taking only the first reachable row (1073). Two are committed as tests.
* **Per-obligation scoring**, 304 keys × {real, ieee}: **81 obligations moved,
  every one UNKNOWN → definite, every one agreeing with an executing oracle,
  0 wrong moves**; 25 out-of-bounds findings where the baseline emitted 0.
* **`corpus/supply`**: all 20 harnesses **byte-identical** after normalising
  solver timings. The round buys nothing there and costs nothing — those
  harnesses contain no dynamic indexing.

## Covered, and declined

**Covered.** Any rank; any `slice_sizes`, contiguous or not (the primitive
takes one contiguous window per axis and the transfer follows it); multiple
independent start indices; the `u[i, j]` multi-axis form; negative indices via
the upstream normalisation; the covered leading-axis gather row form with a
range-valued index; point starts (which reproduce the exact slice).

**Declined, each with a named reason.** A start straddling the legal window; a
start range disjoint from it (a finding); non-integral or unbounded index
intervals — this layer is handed bounds with no dtype and never rounds
`[0.5, 5.5]` inward to the integers it contains; an index dtype too narrow to
hold the axis' bound; a hull whose enumeration would exceed the work budget
(degrade-don't-hang: the budget can cost precision, never soundness).

**Untouched.** Gather geometries outside the covered row form — batching
dimensions, multi-column index vectors, non-leading collapsed axes, and the
`vmap` form (`offset_dims=(1,)`, `collapsed_slice_dims=()`), which is a
different geometry and declines exactly as before. `scatter` and `scatter-add`
keep their static-index rows; only their *definite* out-of-range message was
reclassified as a finding. No SMT emission row: `dynamic_slice` and
`dynamic_update_slice` are absent from `obligation._SUPPORTED`, so an
obligation reaching one cannot escalate to a solver.

**Recorded as a limitation.** Under `semantics="ieee"`, jnp's from-the-end
normalisation declines at its integer `add` (endpoint arithmetic there is
binary64-only) before the row is reached, so the ieee leg buys nothing for
jnp-spelled dynamic indexing. The row itself is sound as-is there, pinned by
binding the primitive directly.

**Assumed, and unconfirmed.** XLA computes an index's out-of-bounds
comparison in the *index's own element type*, where a bound that does not fit
wraps — the hazard `_scatter_index_dtype_covers` exists for. Probing jax
0.11.0's `dynamic_slice` with an `int8` start over operand lengths
100/127/128/129/200 did **not** exhibit it. The gate is applied anyway: every
index dtype jnp's own indexing produces is `int32`/`int64`, for which any
length jax can allocate fits, so refusing the unconfirmed case is free.

**Interaction with the open integer-literal wrap defect, stated.** The
normalisation computes `i + n` on a wrappable dtype. It cannot wrap in a way
that matters: the `select_n` takes that branch only when `i < 0`, and
`i + n > INT_MAX` with `i < 0` is impossible for any `n` an array length can
be. Where a declared index interval is wide enough that stelling's own
integer `add` declines, the index arrives ⊤ and this round declines with it —
sound, and the standing behaviour.
