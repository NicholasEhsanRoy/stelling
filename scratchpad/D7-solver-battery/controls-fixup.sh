# Positive controls for the gates added in the D7 fixup, 2026-08-23.
#
# Every one of these mutations left the module GREEN before the fixup. Each is
# applied, the named gate is run, and the tree is restored. A control that
# does not go red is a gate that is not there.
#
#   bash scratchpad/D7-solver-battery/controls-fixup.sh
set -u
WT=/home/nick/MSF/stelling-wt/D7
PY=/home/nick/venvs/stelling-jax/bin/python
cd "$WT"
run() { COLUMNS=100 JAX_PLATFORMS=cpu PYTHONPATH=$WT/src timeout 600 $PY \
        -m pytest -q -p no:randomly "tests/test_solver_battery.py::$1" 2>&1 | tail -1; }
snap() { cp "$1" "/tmp/ctl_$(basename $1).bak"; }
rest() { mv "/tmp/ctl_$(basename $1).bak" "$1"; \
         find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; }

PAGE=docs/choosing-a-solver-backend.md
TOOL=tools/solver_battery.py
TEST=tests/test_solver_battery.py

echo "--- C12: the literal reading's cells reversed to the page's direction"
echo "         (the headline finding, INVERTED on the page)"
snap $PAGE
python3 - <<'PYX'
import pathlib
p = pathlib.Path("docs/choosing-a-solver-backend.md"); s = p.read_text()
old = "| unsat, 4.4–4.6 s | **UNKNOWN** (16.0 s wall) |"
assert s.count(old) == 1
p.write_text(s.replace(old, "| **UNKNOWN** (timeout) | unsat, 166–175 ms |"))
PYX
run test_the_three_readings_table_still_shows_what_the_prose_says
rest $PAGE

echo "--- C13: row 7's contested string emptied"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
head = '''        contested=(
            "the page's row says z3 TIMED OUT and cvc5 answered in 166-175 ms. "'''
i = s.index(head)
j = s.index('),\n    ),\n    Row(\n        n=8,', i)
p.write_text(s[:i] + '        contested="",\n' + s[j + len('),\n'):])
PYX
run test_the_page_s_marks_are_the_tool_s_refusals
rest $TOOL

echo "--- C14: Motzkin's -3 -> -2  (no longer Motzkin; still nonneg, still"
echo "         degree 6, still 2 vars, still unsat)"
snap $TOOL
sed -i 's/return x2 \* x2 \* y2 + x2 \* y2 \* y2 - 3.0 \* x2 \* y2 + 1.0/return x2 * x2 * y2 + x2 * y2 * y2 - 2.0 * x2 * y2 + 1.0/' $TOOL
run test_the_named_objects_are_the_objects_their_labels_name
rest $TOOL

echo "--- C15: AM-GM's 2xy -> 1.5xy  (no longer AM-GM; still true)"
snap $TOOL
sed -i 's/return x \* x + y \* y, 2.0 \* x \* y/return x * x + y * y, 1.5 * x * y/' $TOOL
run test_the_named_objects_are_the_objects_their_labels_name
rest $TOOL

echo "--- C16: a PUBLISHED cell on the page edited"
snap $PAGE
sed -i 's/| unsat, 78–112 ms | unsat, 8–9 ms |/| unsat, 999–999 ms | unsat, 8–9 ms |/' $PAGE
run test_the_battery_is_the_page_s_table_row_for_row
rest $PAGE

echo "--- C17: the three-readings table's hand-copy of row 7 edited"
snap $PAGE
sed -i 's/| \*\*the row above, as published\*\* | \*\*UNKNOWN\*\* (timeout) | unsat, 166–175 ms |/| **the row above, as published** | **UNKNOWN** (timeout) | unsat, 999–999 ms |/' $PAGE
run test_the_three_readings_table_still_shows_what_the_prose_says
rest $PAGE

echo "--- C18: a second ten-row copy re-added to the page"
snap $PAGE
python3 - <<'PYX'
import pathlib
p = pathlib.Path("docs/choosing-a-solver-backend.md"); s = p.read_text()
anchor = "**What held.**"
copy = ("| row (same label) | fragment | both | z3 alone | cvc5 alone |\n"
        "|---|---|---|---|---|\n"
        "| scalar, linear | `QF_LRA` | unsat, 73–80 ms | unsat, 6–9 ms | "
        "unsat, 71–78 ms |\n\n")
assert s.count(anchor) == 1
p.write_text(s.replace(anchor, copy + anchor))
PYX
run test_the_page_carries_no_second_copy_of_the_ten_row_table
rest $PAGE

echo "--- C19: the page's own instruction inverted"
snap $PAGE
sed -i 's/Read the ten rows for their \*direction\*, not their milliseconds/Read the ten rows for their milliseconds, not their *direction*/' $PAGE
run test_the_page_tells_the_reader_how_to_re_derive_the_table
rest $PAGE

echo "--- C20: a row's reconstruction grade promoted on the page only"
snap $PAGE
sed -i 's/| unsat, 166–175 ms | unsupported ‡ |/| unsat, 166–175 ms | reconstructed |/' $PAGE
run test_the_battery_is_the_page_s_table_row_for_row
rest $PAGE

echo "--- C21: a blank fragment called a DISAGREEING one again"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
old = '''        if not m.fragment:
            agree = "   <- NOT MEASURED"
        elif m.fragment != r.fragment:'''
new = '''        if False:
            agree = "   <- NOT MEASURED"
        elif m.fragment != r.fragment:'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run test_it_runs_with_no_jax_and_names_jax_as_what_is_missing
rest $TOOL

echo "--- C22: a repeat that RAISED counted as an attempt again"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
old = '''        return bool(self.ms) and not any(
            o in ("error", "not measured") for o in self.outcomes)'''
new = '''        return self.outcome not in ("not measured", "")'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run test_a_repeat_that_raised_is_not_a_finding_that_did_not_hold
rest $TOOL

echo "--- C22b: the same regression in its original spelling"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
old = '''        return bool(self.ms) and not any(
            o in ("error", "not measured") for o in self.outcomes)'''
new = '''        return bool(self.ms) and not any(
            o in ("not measured",) for o in self.outcomes)'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run test_a_repeat_that_raised_is_not_a_finding_that_did_not_hold
rest $TOOL

echo "--- C23: backend presence decided from the WHEEL alone again"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
old = '''        cvc5_ok = _optional.available("cvc5") or cvc5_bin is not None'''
new = '''        cvc5_ok = _optional.available("cvc5")'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run test_the_environment_probe_asks_the_same_question_stelling_answers
rest $TOOL

echo "--- C24: tools/ put back on sys.path at module scope"
snap $TEST
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tests/test_solver_battery.py"); s = p.read_text()
old = "    spec = importlib.util.spec_from_file_location(\n        _PRIVATE_MODULE_NAME,"
new = ("    sys.path.insert(0, str(TOOLS))  # PLANTED\n"
       "    spec = importlib.util.spec_from_file_location(\n"
       "        _PRIVATE_MODULE_NAME,")
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run test_loading_the_tool_leaves_nothing_behind_in_this_session
rest $TEST

echo "--- clean tree again:"
COLUMNS=100 JAX_PLATFORMS=cpu PYTHONPATH=$WT/src timeout 900 $PY -m pytest -q \
  -p no:randomly tests/test_solver_battery.py 2>&1 | tail -1
