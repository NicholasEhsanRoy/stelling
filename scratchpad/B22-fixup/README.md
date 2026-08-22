<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# B22 fixup — measurement drivers, mutation harness, and run logs

Rescued from `/tmp` at a forced shutdown. Everything here is EVIDENCE, not
shipped code. Run any driver as:

    cd <worktree>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=<0|1> PYTHONPATH=<worktree>/src \
      /home/nick/venvs/<stelling-jax|stelling-jax010>/bin/python \
      scratchpad/B22-fixup/<driver>.py

| driver | what it answers |
|---|---|
| `s1.py`  | does a float value that overflows warn? (record mode, per case) |
| `s1b.py` | the same under `simplefilter("error")`, EAGER **and** inside `jit` |
| `s1c.py` | HOST vs DEVICE — where the residue actually is |
| `s1d.py` | where the `RuntimeWarning` is raised FROM (frames) |
| `s2.py`  | the float8 formats: disarmed silence, image, armed refusal |
| `s2b.py` | EVERY float dtype `jax.numpy` exposes, derived not declared |
| `s2c.py` | the eight quiet formats, eager vs traced, under `error` |
| `s3.py`  | is `x_f16 <= 100000` silent? eager / make_jaxpr / jit |
| `g6.py`  | StableHLO: `loc(` counts and armed-vs-disarmed, both spellings |
| `which.py` | which HOST case is silent at x64=1 (found the mis-classified one) |

`mutate.sh` is the mutation harness: `source mutate.sh` then
`run <label> <file> <python-snippet> <pytest -k expr>`. It asserts the mutation
APPLIED (`git diff --quiet` after) before believing any result, and restores
with `git checkout --` afterwards.

`cells3.sh` runs the full suite in the three non-default cells.

## MEASURED FIGURES (all four cells = jax 0.10.2 / 0.11.0 x x64 off/on)

### S1 — the silence axis

Under `warnings.simplefilter("error")`. `RuntimeWarning: overflow encountered
in cast`, raised from jax's host-side `lax._convert_element_type` numpy cast.

| case | 010/x0 | 010/x1 | 011/x0 | 011/x1 |
|---|---|---|---|---|
| `jnp.full((2,), 1e300, float32)` | WARNS | WARNS | WARNS | WARNS |
| `jnp.full((2,), 70000.0, float16)` | WARNS | WARNS | WARNS | WARNS |
| `jnp.float16(70000.0)` | WARNS | WARNS | WARNS | WARNS |
| `jnp.full((2,), 100000, float16)` | WARNS | WARNS | WARNS | WARNS |
| `jnp.array([1e300], float32)` | WARNS | WARNS | WARNS | WARNS |
| `x_f32 + 1e300` EAGER | WARNS | SILENT | WARNS | SILENT |
| `jnp.asarray([1e300,1e300]).astype(float32)` EAGER | WARNS | SILENT | WARNS | SILENT |
| `x_f16 + 70000.0` EAGER | SILENT | SILENT | SILENT | SILENT |
| **all five float cases inside `jit`** | WARNS | WARNS | WARNS | WARNS |
| INT `jnp.full((), 256, int8)` -> 0 | SILENT | SILENT | SILENT | SILENT |
| INT `jnp.full((3,), 40000, int16)` -> -25536 | SILENT | SILENT | SILENT | SILENT |
| INT `x_int16 + 40000`, `jit(x_int8 + 256)` | SILENT | SILENT | SILENT | SILENT |

Genuinely silent (DEVICE), identical in all four cells: `a*a`, `a**2`,
`jit(a*a)` on float32 `[1e30,1e30]`; `jnp.exp(float32 1000.0)`;
`a.astype(float16)`; `lax.convert_element_type(a, float16)`. All `inf`, 0 fires
from all three instruments, and `-W error::RuntimeWarning` does not reach them.

### S2 — the quiet float formats (all four cells identical)

`propagate._FLOAT_FORMATS` = float16, bfloat16, float32, float64 (the
VERIFIER's catalogue). `prop_guard` has none — it asks `ml_dtypes.finfo`.

| dtype | max | written | runs as | eager | traced |
|---|---|---|---|---|---|
| `float16` | 65504 | `x <= 100000` | inf | silent | **WARNS** |
| `float8_e3m4` | 15.5 | `x <= 33` | inf | silent | silent |
| `float8_e4m3` | 240 | `x <= 483` | inf | silent | silent |
| `float8_e5m2` | 57344 | `x <= 114691` | inf | silent | silent |
| `float8_e4m3fn` | 448 | `x <= 899` | **nan** | silent | silent |
| `float8_e4m3b11fnuz` | 30 | `x <= 63` | **nan** | silent | silent |
| `float8_e4m3fnuz` | 240 | `x <= 483` | **nan** | silent | silent |
| `float8_e5m2fnuz` | 57344 | `x <= 114691` | **nan** | silent | silent |

All eight refused `overflows-float` armed. The four `nan` rows invert the
comparison to `[False, False, False]`. `float8_e8m0fnu` (max 1.7e38) cannot
lose an int64. `float4_e2m1fn` SATURATES (image 6.0) so it is `inexact`, not
`overflows-float`. `float6_e2m3fn` / `float6_e3m2fn` cannot be built at all —
`JaxRuntimeError: Invalid XLA PrimitiveType`.

### G6 — StableHLO (all four cells)

`jit(...).lower().as_text()`: len 3646, **0** `loc(`, IDENTICAL armed==disarmed.
`as_text(debug_info=True)`: len 9766, **144** `loc(`, **DIFFERS**.
Control: 5 guard checks during the lowering, so the wrapper is live.
NOTE: `debug_info=True` records the CALLER's line number — two call sites
differ with nothing armed. Compare from ONE source line only.

### Suite results

* full suite, jax 0.11.0 x64=0, at `cbce692`: **4568 passed, 10 skipped**, exit 0
* full suite, jax 0.10.2 x64=0, at `502d7e6`: **4568 passed, 10 skipped**, exit 0
* full suite, 0.10.2 x64=1 and 0.11.0 x64=1: **NOT COMPLETED** (shutdown) —
  see "next step" in the handover
* changed modules (`test_narrowing_perimeter` + `test_tripwire_arm` +
  `test_doc_examples` + `test_tripwire_record`), all four cells at `c53ff6b`:
  328 / 327+1skip / 328 / 327+1skip, all green
* `pytest -p stelling.overflow --stelling-overflow=require` on
  `test_narrowing_perimeter.py`, all four cells at `cbce692`: **armed**, exit 0,
  **0 PARTIAL**, no `_state_guard` error. (Before the fix: `NOT ARMED
  [detached]`, PARTIAL x2, exit 1, ERROR at teardown.)
* `pytest --stelling-narrowing-perimeter=error` whole suite, jax 0.11.0 x64=0,
  at `cbce692`: **4568 passed, 10 skipped**, exit 0, `1473 integer literal(s)`
  checked, `15 narrowing(s) PERMITTED at 9 site(s)`
* `pytest -W error::RuntimeWarning tests/test_narrowing_perimeter.py`: 70 passed
  (the new gate handles its warnings explicitly rather than filtering them)

### Mutations driven — every fix watched failing first

| # | mutation | result |
|---|---|---|
| M1 | a HOST case declared SILENT | REDDENS |
| M2 | a DEVICE case declared WARNING | REDDENS |
| M3 | page drops the `-W error::RuntimeWarning` remedy | REDDENS |
| M4 | page drops one float8 row | REDDENS |
| M5 | `QUIET_FLOAT_FORMATS` reverts to the borrowed catalogue | REDDENS |
| G1 | page program -> `x + 300` | REDDENS (was GREEN before the fix) |
| G2 | artefact row flipped between halves | REDDENS (was GREEN) |
| G3a | what-runs cell `-25536` -> `-255360` | REDDENS (was GREEN) |
| G3b | comparison cell `2147483648.0` -> `.09` | REDDENS (was GREEN) |
| G4 | the SECOND `x_f16 <= 100000` rots to `999` | REDDENS (was GREEN) |
| G6a | debug_info row moved to the identical half | REDDENS |
| G6b | debug_info row deleted from the page | REDDENS |
| M10 | `report.py` sentence reverted | REDDENS |
| M11 | x64-dependent pair declared unconditional | GREEN at x64=0, **REDDENS at x64=1** |
| M12 | `x_f16 + 70000.0` eager declared WARNING | REDDENS |
| G5 | `borrowed_tripwire` reverted to unconditional disarm | dial-on session exit **1**, `NOT ARMED [detached]`, PARTIAL x2, `_state_guard` ERROR |
| — | `_LOUDER_THAN_THE_PAGE` definition deleted | float gate GREEN, new reachability test REDDENS |

G1's true answers, driven: jaxpr `{ lambda ; a:i8[1]. let b:i8[1] = add a
44:i8[] in (b,) }`, result `[-112, 94, 34]`.

`test_narrowing_perimeter.py` sorts **72nd of the 149 files** `pytest
--collect-only -q` names here; `find tests -name 'test_*.py'` counts **155**.
