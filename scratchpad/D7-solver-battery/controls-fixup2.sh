# Positive controls for the gates added in the D7 fixup, ROUND 2, 2026-08-23.
#
# Every mutation below left the module GREEN at 943b9c6 (38 passed) — six of
# them were driven against a worktree at that commit and are recorded in the
# README. Each is applied here, the named gate is run, and the tree restored.
# A control that does not go red is a gate that is not there.
#
#   bash scratchpad/D7-solver-battery/controls-fixup2.sh
set -u
WT=/home/nick/MSF/stelling-wt/D7
PY=/home/nick/venvs/stelling-jax/bin/python
cd "$WT"
run() { COLUMNS=100 JAX_PLATFORMS=cpu PYTHONPATH=$WT/src timeout 600 $PY \
        -m pytest -q -p no:randomly "tests/test_solver_battery.py::$1" 2>&1 | tail -1; }
snap() { cp "$1" "/tmp/ctl2_$(basename $1).bak"; }
rest() { mv "/tmp/ctl2_$(basename $1).bak" "$1"; \
         find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; }

PAGE=docs/choosing-a-solver-backend.md
TOOL=tools/solver_battery.py
GRADEGATE=test_the_page_states_every_grade_s_meaning_in_the_tool_s_own_words
FIGGATE=test_every_quoted_sweep_figure_is_in_the_transcript_it_is_attributed_to
DERIVED=test_a_contested_row_s_grade_is_DERIVED_from_its_readings

echo "--- C25: the page's standalone limit sentence INVERTED into the overclaim"
snap $PAGE
python3 - <<'PYX'
import pathlib
p = pathlib.Path("docs/choosing-a-solver-backend.md"); s = p.read_text()
old = "> `reconstructed` does NOT mean `this battery reproduced the published"
new = "> `reconstructed` MEANS `this battery REPRODUCED the published"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run $GRADEGATE
rest $PAGE

echo "--- C26: RECONSTRUCTED_IS_NOT_A_REPRODUCTION inverted in the tool"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
old = '''    "`reconstructed` does NOT mean `this battery reproduced the published "'''
new = '''    "`reconstructed` MEANS `this battery REPRODUCED the published "'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run $GRADEGATE
rest $TOOL

echo "--- C27: one word changed in the page's copy of a grade's meaning"
snap $PAGE
python3 - <<'PYX'
import pathlib
p = pathlib.Path("docs/choosing-a-solver-backend.md"); s = p.read_text()
old = "| `direction only` | 9, 10 | the readings agree on WHICH BACKEND WINS"
new = "| `direction only` | 9, 10 | the readings disagree on WHICH BACKEND WINS"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run $GRADEGATE
rest $PAGE

echo '--- C28: the page grade table drops row 2 from `outcome only`'
echo "         (the ten-row table still agrees, so only the new gate sees it)"
snap $PAGE
python3 - <<'PYX'
import pathlib
p = pathlib.Path("docs/choosing-a-solver-backend.md"); s = p.read_text()
old = "| `outcome only` | 2, 5 |"
new = "| `outcome only` | 5 |"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run $GRADEGATE
rest $PAGE

echo '--- C29: row 5 promoted back to `reconstructed` in the tool alone'
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
old = '''        page_cvc5="unsat, 81–83 ms",
        grade=GRADE_OUTCOME_ONLY,'''
new = '''        page_cvc5="unsat, 81–83 ms",
        grade=GRADE_RECONSTRUCTED,'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run test_the_battery_is_the_page_s_table_row_for_row
rest $TOOL

echo "--- C30: the page's 1.22x re-typed as 1.02x"
snap $PAGE
python3 - <<'PYX'
import pathlib
p = pathlib.Path("docs/choosing-a-solver-backend.md"); s = p.read_text()
assert s.count("1.22x") == 2
p.write_text(s.replace("1.22x", "1.02x"))
PYX
run $FIGGATE
rest $PAGE

echo "--- C31: the tool's 17.82x re-typed as 1.82x (the page still says 17.82x)"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
assert s.count("17.82x") == 3
p.write_text(s.replace("17.82x", "1.82x"))
PYX
run $FIGGATE
rest $TOOL

echo "--- C32: solvers._escalate, which never existed, put back on the page"
snap $PAGE
python3 - <<'PYX'
import pathlib
p = pathlib.Path("docs/choosing-a-solver-backend.md"); s = p.read_text()
old = "`solvers._dispatch_obligation`, which `solvers.escalate` reaches for each"
new = "`solvers._escalate`, which `solvers.escalate` reaches for each"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run test_every_solver_symbol_these_two_cite_exists
rest $PAGE

echo "--- C33: INTERVAL_UNDECIDED dropped from row 4 again"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
old = '''                "row keeps the strongest grade and rows 2 and 5 do not",
                INTERVAL_UNDECIDED, _TIMED),'''
new = '''                "row keeps the strongest grade and rows 2 and 5 do not",
                _TIMED),'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run test_every_row_says_what_the_page_left_open
rest $TOOL

echo "--- C34: row 8's two splitting readings flipped so none puts z3 ahead"
echo "         (the exact shape of the grade row 8 did not meet)"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
for old, new in (
    ('''    Reading(8, "sum-of-squares", True, True, "z3",''',
     '''    Reading(8, "sum-of-squares", True, True, "cvc5",'''),
    ('''    Reading(8, "cancellation", True, True, "z3",''',
     '''    Reading(8, "cancellation", True, True, "cvc5",'''),
):
    assert s.count(old) == 1
    s = s.replace(old, new)
p.write_text(s)
PYX
run $DERIVED
rest $TOOL

echo "--- C35: row 9 given a reading that REVERSES it, still graded direction only"
snap $TOOL
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
old = '''    Reading(9, "one-variable: x^10 >= 0", True, True, "z3",'''
new = '''    Reading(9, "one-variable: x^10 >= 0", False, True, "cvc5",'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PYX
run $DERIVED
rest $TOOL

echo '--- C36: row 10 derived factor typed back as `a hundred to three hundred`'
snap $PAGE
python3 - <<'PYX'
import pathlib
p = pathlib.Path("docs/choosing-a-solver-backend.md"); s = p.read_text()
# BOTH places the page states it -- the paragraph that publishes the factor
# and the row-9 note that compares row 10 to it.
assert s.count("98x\u2013351x") == 2
p.write_text(s.replace("98x\u2013351x", "a hundred to three hundred"))
PYX
run test_the_clock_gap_the_page_states_is_the_one_its_transcripts_show
rest $PAGE

echo "--- baseline: unmutated"
COLUMNS=100 JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY -m pytest -q -p no:randomly \
  tests/test_solver_battery.py 2>&1 | tail -1
