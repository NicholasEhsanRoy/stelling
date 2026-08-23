source /home/nick/MSF/stelling-wt/B22/scratchpad/B22-fixup2/mutate.sh
P=docs/overflow-tripwire.md
T=tests/test_narrowing_perimeter.py
PATHS="tests/test_narrowing_perimeter.py tests/test_doc_examples.py tests/test_prose_hygiene.py"

run "N1 float8 table: e4m3fn runs-as nan -> inf" "$P" \
  't = t.replace("| `float8_e4m3fn` | 448 | `x <= 899` | **`nan`** |", "| `float8_e4m3fn` | 448 | `x <= 899` | `inf` |")' "$PATHS"
run "N2 float8 table: float16 traced **warns** -> silent" "$P" \
  't = t.replace("| `float16` | 65504 | `x <= 100000` | `inf` | silent | **warns** |", "| `float16` | 65504 | `x <= 100000` | `inf` | silent | silent |")' "$PATHS"
run "N3 page: Eight -> Nine" "$P" \
  't = t.replace("**Eight of `jax.numpy`", "**Nine of `jax.numpy`")' "$PATHS"
run "N4 page: device-silent list rewritten" "$P" \
  't = t.replace("`a * a` and `a ** 2` on a `float32` array of `1e30`, `jnp.exp` of a\n  `float32` `1000.0`, `a.astype(jnp.float16)` and\n  `lax.convert_element_type(a, jnp.float16)` on that same array, and\n  `x_f16 + 70000.0` run EAGERLY", "`jnp.sin` and `jnp.cosh` on a `float64` array of `1e30`, `jnp.log` of a\n  `float32` `1000.0`, `a.astype(jnp.bfloat16)` and\n  `lax.bitcast_convert_type(a, jnp.float16)` on that same array, and\n  `x_f16 * 70000.0` run under `jit`")' "$PATHS"
run "N5 report._suggestions door list" "src/stelling/_tripwire/report.py" \
  't = t.replace("(jnp.array, jnp.asarray, jnp.int8) raise ", "(jnp.full, jnp.full_like, jnp.where) raise ")' "tests/test_tripwire_arm.py"
run "N7 page: axis sentence INVERTED" "$P" \
  't = t.replace("**The axis is WHERE the narrowing happens, not how the line is spelled**", "**The axis is HOW THE LINE IS SPELLED, not where the narrowing happens**")' "$PATHS"
run "N8 page: -W error DOES reach the device residue" "$P" \
  't = t.replace("but it is the boundary, and `-W error::RuntimeWarning` does not\n  reach it.", "but it is the boundary, and `-W error::RuntimeWarning` DOES\n  reach it.")' "$PATHS"
run "N9 page: remedy sentence inverted" "$P" \
  't = t.replace("`-W error::RuntimeWarning` covers whatever\n  numpy touched and nothing else.", "`-W error::RuntimeWarning` covers nothing\n  numpy touched and everything else.")' "$PATHS"
run "N10 page: float16 the only one traced -> none" "$P" \
  't = t.replace("**traced, `float16` is the one format where\n  numpy'"'"'s host cast warns**", "**traced, NO format is one where\n  numpy'"'"'s host cast warns**")' "$PATHS"
run "N6 page: HOST bullet dtypes changed to bfloat16" "$P" \
  't = t.replace("`jnp.full((2,), 1e300, jnp.float32)`, `jnp.full((2,), 70000.0, jnp.float16)`,", "`jnp.full((2,), 1e300, jnp.bfloat16)`, `jnp.full((2,), 70000.0, jnp.float8_e5m2)`,")' "$PATHS"
run "N11 page: dial-on 4565 -> 9999" "$P" \
  't = t.replace("4565 passed, 10 skipped", "9999 passed, 10 skipped")' "$PATHS"
run "N12 149 -> 3333 in test and CHANGELOG" "$T CHANGELOG.md" \
  't = t.replace("149 files", "3333 files")' "$PATHS"
