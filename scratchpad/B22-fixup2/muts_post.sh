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

# ---- the CHANGELOG's two counts, and the case-6 operand ---------------------
run "N17 CHANGELOG: five warn eagerly -> four" "CHANGELOG.md" \
  't = t.replace("**five warn eagerly with x64\n  off, three with x64 on**", "**four warn eagerly with x64\n  off, three with x64 on**")' "$PATHS"
run "N18 CHANGELOG: five of the six -> all of them" "CHANGELOG.md" \
  't = t.replace("**five of the six warn inside `jit`**", "**all of them warn inside `jit`**")' "$PATHS"

# ---- the dial-on figures. ONE artefact reddens; BOTH together do not, which
# ---- is the declared limit: nothing holds a whole-suite figure CURRENT.
run "N11a page dial-on 4575 -> 9999 (page only)" "$P" \
  't = t.replace("4575 passed, 10 skipped", "9999 passed, 10 skipped")' "$PATHS"
run "N11b CHANGELOG dial-on 4575 -> 9999 (changelog only)" "CHANGELOG.md" \
  't = t.replace("**4575 passed, 10\n  skipped**", "**9999 passed, 10\n  skipped**")' "$PATHS"
run "N11c BOTH changed together -- STAYS GREEN, and that is the limit" "$P CHANGELOG.md" \
  't = t.replace("4575 passed, 10\n  skipped", "9999 passed, 10\n  skipped").replace("4575 passed, 10 skipped", "9999 passed, 10 skipped")' "$PATHS"
run "N11d page permitted 15 at 9 -> 15 at 7 (page only)" "$P" \
  't = t.replace("region, at 9 site(s)\n4575", "region, at 7 site(s)\n4575")' "$PATHS"

# ---- CODE-SIDE CONTROLS: make the claim false and the gate must notice ------
run "C1 code: HOST_DOOR_SILENT gains float16 (partition false)" "$T" \
  't = t.replace("HOST_DOOR_SILENT = (\n    \"bfloat16\",", "HOST_DOOR_SILENT = (\n    \"float16\",\n    \"bfloat16\",")' "$PATHS"
run "C2 code: a construction door dropped from the drive" "$T" \
  't = t.replace("    (\"jnp.<dt>(LIT)\", lambda dt, lit: dt(lit)),\n", "")' "$PATHS"
run "C3 code: FLOAT_OVERFLOW_DEVICE_SILENT loses a case" "$T" \
  't = t.replace("    (\"a ** 2 on float32 1e30\", lambda: _big_f32() ** 2, \"a ** 2\"),\n", "")' "$PATHS"
run "C4 code: QUIET_FLOAT_FORMATS loses float8_e3m4" "$T" \
  't = t.replace("    (\"float8_e3m4\",         15.5,        33, float(\"inf\"), False),\n", "")' "$PATHS"
run "C5 code: case 6 operand built INSIDE the window" "$T" \
  't = t.replace("lambda: jax.jit(lambda a: a.astype(jnp.float32))(_six_operand())", "lambda: jax.jit(lambda a: a.astype(jnp.float32))(jnp.asarray([1e300, 1e300]))")' "$PATHS"
run "N5 report._suggestions door list" "src/stelling/_tripwire/report.py" \
  't = t.replace("(jnp.array, jnp.asarray, jnp.int8) raise ", "(jnp.full, jnp.full_like, jnp.where) raise ")' "tests/test_tripwire_arm.py"
