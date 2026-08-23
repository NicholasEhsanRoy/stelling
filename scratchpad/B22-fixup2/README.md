<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# B22 fixup, round 2 — the axis is the TARGET FORMAT

Everything here is EVIDENCE, not shipped code. Run any driver as:

    cd <worktree>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=<0|1> PYTHONPATH=<worktree>/src \
      /home/nick/venvs/<stelling-jax|stelling-jax010>/bin/python \
      scratchpad/B22-fixup2/<driver>.py

and verify `stelling.__file__` resolves into that worktree first.

| driver | what it answers |
|---|---|
| `r1_axis.py`  | the three construction doors x five dtypes, jax and jax-free |
| `r2_where.py` | is the quiet narrowing on the HOST? (`make_jaxpr`) + which numpy API |
| `r3_doors.py` | EVERY float dtype `jax.numpy` exposes x every door, derived not declared |
| `r4_mech.py`  | the mechanism: target format, or a `float32` intermediate? |
| `r5_counts.py`| the CHANGELOG's two counts over the six cases the pre-fix page named |

`mutate.sh` is the mutation harness (`source mutate.sh`, then
`run <label> <files> <python-snippet> <pytest paths> [venv] [x64]`). It asserts
the mutation APPLIED before believing any result and restores afterwards.
`muts_pre.sh` is the battery as it stood BEFORE this round's repairs — twelve
mutations, twelve green. `muts_post.sh` is the same battery afterwards.

The four-cell full-suite driver is `../B22-fixup/cells3.sh`, repaired in this
round: as committed at `725edc3` its `WT=` pointed at a deleted worktree and it
ran pytest zero times.

`logs/` holds the raw output of every whole-suite run this round drove
(`.txt` because `.gitignore` drops `*.log`): the four cells, the dial-on
sessions, and the collection order the `148 files / 72nd` figures come from.

## THE FINDING — host versus device does not partition it

`docs/overflow-tripwire.md` stated, at `:154-156`, that *"where the narrowing
is done ON THE HOST, by numpy, the cast emits `RuntimeWarning`"*, and drew the
reader's remedy from it at `:186-187`. **Measured, in all four cells:**

    jnp.full((2,), 1e300, jnp.float16)     WARNS
    jnp.full((2,), 1e300, jnp.float32)     WARNS
    jnp.full((2,), 1e300, jnp.bfloat16)    silent -> inf
    jnp.array([1e300],    jnp.bfloat16)    silent -> inf
    jnp.bfloat16(1e300)                    silent -> inf

and it IS a host narrowing: `make_jaxpr` holds `inf:bf16[]` before any XLA
program exists, exactly as the `float16` route holds `inf:f16[]`.

**The axis is the TARGET FORMAT.** The warning is raised by numpy's own
floating-point machinery, which knows only the formats numpy implements
itself. `hasattr(numpy, name)` is the discriminator, and it is not
`dtype.kind`: `np.dtype(ml_dtypes.float8_e5m2).kind` is `'f'` and it behaves
like the extension formats.

| | numpy's own | `ml_dtypes` |
|---|---|---|
| formats `jax.numpy` exposes | `float16`, `float32`, `float64` | the other **twelve** |
| `jnp.full` / `jnp.array` / `jnp.<dt>` | **WARNS** | silent |
| `.astype` from `float32` | **WARNS** | silent |
| `.astype` from `float64` past `float32`'s range | **WARNS** | **WARNS** — about the `float32` intermediate |

**The decisive control, with no jax in it** (`r4_mech.py`): one numpy cast
loop, one `float32` source array of `1e30`.

    np.float32([1e30]).astype(float16)         WARNS
    np.float32([1e30]).astype(float8_e5m2)     silent -> inf
    np.float32([1e30]).astype(float8_e4m3fn)   silent -> nan

Same call, same source; the two silent ones lost four more orders of
magnitude. Identical on `ml_dtypes` 0.5.4 (jax 0.10.2) and 0.6.0 (jax 0.11.0).

**And the `.astype` warnings on extension formats are not about the target at
all**: they appear exactly when the source also overflows `float32`. Measured:
`np.asarray(5.73e6).astype(float8_e5m2)` (max 57344) is silent;
`np.asarray(1e300).astype(float8_e5m2)` warns.

## The counts the CHANGELOG had wrong (`r5_counts.py`)

Over the six cases the pre-fix page named, under `simplefilter("error")`:

| | 010/x0 | 010/x1 | 011/x0 | 011/x1 |
|---|---|---|---|---|
| warn EAGERLY | **5** | **3** | **5** | **3** |
| warn inside `jit` | **5** | **5** | **5** | **5** |

The entry said *"four warn eagerly"* (right in no cell) *"identical in all
four cells"* (false of the eager half) and *"all of them warn inside `jit`"*
(false: case 6, `jit(a.astype(jnp.float32))` on `[1e300, 1e300]` with the
operand built outside the window, is silent in all four).

## The float6 trap (`jnp.zeros` is not buildability)

| | jax 0.10.2 | jax 0.11.0 |
|---|---|---|
| `jnp.zeros((3,), jnp.float6_e2m3fn)` | `JaxRuntimeError: Invalid XLA PrimitiveType: 36` | **succeeds** |
| the first operation on it | — | `JaxRuntimeError: RET_CHECK ... Invalid bitcast i6 to i8` |
| `jnp.zeros((3,), jnp.float4_e2m1fn)` | succeeds, ops fine | succeeds, ops fine |

The quiet-format gate used `jnp.zeros` as its "can be driven" proxy. It now
classifies from `ml_dtypes` arithmetic — which needs no jax and answers in
every cell — and separately drives one real operation per declared format.
