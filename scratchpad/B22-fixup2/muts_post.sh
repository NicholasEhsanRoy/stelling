source /home/nick/MSF/stelling-wt/B22/scratchpad/B22-fixup2/mutate.sh
P=docs/overflow-tripwire.md
T=tests/test_narrowing_perimeter.py
PATHS="tests/test_narrowing_perimeter.py"

# ---- S3: the what-runs repair, which was dead code -------------------------
run "S3a _WHAT_RUNS_TRACED key never looked up (KeyError if reached)" "$T" \
  't = t.replace("    \"x_f32 <= 2**31 - 1\": (\n        lambda: jax.make_jaxpr", "    \"UNREACHED_KEY\": (\n        lambda: jax.make_jaxpr", 1)' "$PATHS"
run "S3b -25536 -> -25537 in page AND _WHAT_RUNS" "$T $P" \
  't = t.replace("-25536", "-25537")' "$PATHS"
run "S3c 2147483648.0 -> 999.0 in page AND _WHAT_RUNS" "$T $P" \
  't = t.replace("2147483648.0", "999.0")' "$PATHS"

# ---- N1..N12 ---------------------------------------------------------------
run "N1 float8 table: e4m3fn runs-as nan -> inf" "$P" \
  't = t.replace("| `float8_e4m3fn` | 448 | `x <= 899` | **`nan`** |", "| `float8_e4m3fn` | 448 | `x <= 899` | `inf` |")' "$PATHS"
run "N2 float8 table: float16 traced **warns** -> silent" "$P" \
  't = t.replace("| `float16` | 65504 | `x <= 100000` | `inf` | silent | **warns** |", "| `float16` | 65504 | `x <= 100000` | `inf` | silent | silent |")' "$PATHS"
run "N3 page: Eight -> Nine" "$P" \
  't = t.replace("**Eight can\nlose a literal", "**Nine can\nlose a literal")' "$PATHS"
run "N4 page: device-silent list rewritten" "$P" \
  't = t.replace("`a * a` and `a ** 2` on a `float32` array of `1e30`,\n  `jnp.exp` of a `float32` `1000.0`, `a.astype(jnp.float16)` and\n  `lax.convert_element_type(a, jnp.float16)` on that same array, and\n  `x_f16 + 70000.0` run EAGERLY", "`jnp.sin` and `jnp.cosh` on a `float64` array of `1e30`,\n  `jnp.log` of a `float32` `1000.0`, `a.astype(jnp.bfloat16)` and\n  `lax.bitcast_convert_type(a, jnp.float16)` on that same array, and\n  `x_f16 * 70000.0` run under `jit`")' "$PATHS"
run "N7 page: axis row INVERTED (loud row <-> silent row)" "$P" \
  't = t.replace("| on the HOST, into `float16`, `float32` or `float64` |", "| on the HOST, into any of the **other twelve** float formats `jax.numpy` has |", 1)' "$PATHS"
run "N8 page: -W error DOES reach the device residue" "$P" \
  't = t.replace("It does not reach a\ndevice narrowing in any dtype", "It DOES reach a\ndevice narrowing in any dtype")' "$PATHS"
run "N9 page: remedy sentence inverted" "$P" \
  't = t.replace("it catches a HOST narrowing into `float16`,\n`float32` or `float64`, and it catches nothing else.", "it catches a HOST narrowing into `bfloat16`,\n`float8_e5m2` or `float8_e4m3fn`, and it catches nothing else.")' "$PATHS"
run "N9b page: remedy drops the bfloat16 exclusion" "$P" \
  't = t.replace("and it does not reach a host narrowing into\n`bfloat16` or any of the other eleven formats listed above", "and it reaches every host narrowing, into\n`bfloat16` and the other eleven formats listed above too")' "$PATHS"
run "N10 page: float16 the only one traced -> none" "$P" \
  't = t.replace("**traced, `float16` is the one format where\n  numpy'"'"'s host cast warns**", "**traced, NO format is one where\n  numpy'"'"'s host cast warns**")' "$PATHS"
run "N6 page: the twelve-name list loses one" "$P" \
  't = t.replace("`float8_e3m4`, `float8_e4m3`, `float8_e4m3b11fnuz`,\n", "`float8_e4m3`, `float8_e4m3b11fnuz`,\n")' "$PATHS"
run "N6b page: the twelve-name list gains a loud one" "$P" \
  't = t.replace("**The other twelve are** `bfloat16`,", "**The other twelve are** `float32`, `bfloat16`,")' "$PATHS"
run "N12 148 -> 3333 in test AND CHANGELOG" "$T CHANGELOG.md" \
  't = t.replace("148 files", "3333 files")' "$PATHS"
run "N12b rank 72nd -> 70th in test AND CHANGELOG" "$T CHANGELOG.md" \
  't = t.replace("**72nd of the", "**70th of the")' "$PATHS"
run "N13 page: 15 -> 16 concrete float formats" "$P" \
  't = t.replace("`jax.numpy` exposes **15** concrete", "`jax.numpy` exposes **16** concrete")' "$PATHS"
run "N14 page: saturating count three -> two" "$P" \
  't = t.replace("so those **three** are not this table", "so those **two** are not this table")' "$PATHS"
run "N15 page: control flipped (f8_e5m2 warns)" "$P" \
  't = t.replace("`.astype(jnp.float8_e5m2)` is **silent** and gives `inf`", "`.astype(jnp.float8_e5m2)` **warns** and gives `inf`")' "$PATHS"
run "N16 page: bfloat16 construction doors declared loud" "$P" \
  't = t.replace("`jnp.array([1e300], jnp.bfloat16)` and `jnp.bfloat16(1e300)` are each `inf`\n  with nothing raised", "`jnp.array([1e300], jnp.bfloat16)` and `jnp.bfloat16(1e300)` each RAISE\n  a RuntimeWarning")' "$PATHS"
