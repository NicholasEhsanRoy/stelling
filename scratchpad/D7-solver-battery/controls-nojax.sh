set -u
WT=/home/nick/MSF/stelling-wt/D7
PY=/home/nick/venvs/stelling-nojax/bin/python
cd "$WT"
run() { # name, node
  COLUMNS=100 PYTHONPATH=$WT/src timeout 300 $PY -m pytest -q "tests/test_solver_battery.py::$2" 2>&1 | tail -1
}
snap() { cp "$1" "$1.bak"; }
rest() { mv "$1.bak" "$1"; }

echo "--- control 1: mutate a page cell"
snap docs/choosing-a-solver-backend.md
sed -i 's/| scalar, linear | `QF_LRA` | unsat, 78–112 ms |/| scalar, linear | `QF_LRA` | unsat, 78–113 ms |/' docs/choosing-a-solver-backend.md
run c1 test_the_battery_is_the_page_s_table_row_for_row
rest docs/choosing-a-solver-backend.md

echo "--- control 2: empty a row's chosen tuple"
snap tools/solver_battery.py
python3 - <<'PYX'
import pathlib
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
s = s.replace('''        chosen=("the declared box [1, 2]", "the predicate 2x - x >= 1",
                INTERVAL_UNDECIDED, _TIMED),''', '''        chosen=(),''', 1)
p.write_text(s)
PYX
run c2 test_every_row_says_what_the_page_left_open
rest tools/solver_battery.py

echo "--- control 3: strip the variants that contest row 7"
snap tools/solver_battery.py
python3 - <<'PYX'
import pathlib, re
p = pathlib.Path("tools/solver_battery.py"); s = p.read_text()
s = s.replace('''    Variant(7, "sum-of-squares",
            "sum(a^2 + b^2 - 2ab) >= 0 over [-1,1]^16 x [-1,1]^16",
            "wide_sum_of_squares"),
    Variant(7, "cancellation",
            "sum(a*b) - sum(b*a) >= 0 over [-1,1]^16 x [-1,1]^16",
            "wide_cancellation"),
''', '')
p.write_text(s)
PYX
run c3 test_a_contested_row_has_the_alternate_readings_that_contest_it
rest tools/solver_battery.py

echo "--- control 4: import jax at module scope"
snap tools/solver_battery.py
sed -i 's/^import argparse$/import argparse\nimport jax  # PLANTED/' tools/solver_battery.py
run c4 test_the_module_imports_jax_nowhere_at_module_scope
rest tools/solver_battery.py

echo "--- control 5: put a millisecond in a row label"
snap tools/solver_battery.py
snap docs/choosing-a-solver-backend.md
sed -i 's/        name="scalar, linear",/        name="scalar, linear, 8 ms",/' tools/solver_battery.py
run c5 test_no_row_label_carries_a_millisecond
rest tools/solver_battery.py
rest docs/choosing-a-solver-backend.md

echo "--- and the tree is clean again:"
COLUMNS=100 PYTHONPATH=$WT/src timeout 300 $PY -m pytest -q tests/test_solver_battery.py 2>&1 | tail -1
