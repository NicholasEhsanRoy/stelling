set -u
WT=/home/nick/MSF/stelling-wt/D7
PY=/home/nick/venvs/stelling-jax/bin/python
cd "$WT"
run() { COLUMNS=100 PYTHONPATH=$WT/src timeout 600 $PY -m pytest -q "tests/test_solver_battery.py::$1" 2>&1 | tail -1; }
snap() { cp tools/solver_battery.py /tmp/sb_ctl.bak; }
rest() { mv /tmp/sb_ctl.bak tools/solver_battery.py; find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; }

echo "--- C6: a linear row made nonlinear (fragment gate)"
snap
python3 - <<'PYX'
import pathlib
p=pathlib.Path("tools/solver_battery.py"); s=p.read_text()
s=s.replace("        return assert_(2.0 * x - x >= 1.0)","        return assert_(x * x - x >= 0.0)",1)
p.write_text(s)
PYX
run test_the_fragment_column_is_what_the_page_publishes
rest

echo "--- C7: the 64-element row given 63 elements (declared-input gate)"
snap
python3 - <<'PYX'
import pathlib
p=pathlib.Path("tools/solver_battery.py"); s=p.read_text()
s=s.replace('        x = any_array((64,), _f64(), (1.0, 2.0))','        x = any_array((63,), _f64(), (1.0, 2.0))',1)
p.write_text(s)
PYX
run test_each_row_declares_the_variables_its_label_names
rest

echo "--- C8: a row interval propagation decides, so no backend is ever asked"
snap
python3 - <<'PYX'
import pathlib
p=pathlib.Path("tools/solver_battery.py"); s=p.read_text()
s=s.replace("        return assert_(2.0 * x - x >= 1.0)","        return assert_(2.0 * x - x >= -100.0)",1)
p.write_text(s)
PYX
run test_no_row_is_decided_before_a_backend_is_asked
rest

echo "--- C9: the false cubic made true (obligation-answer gate)"
snap
python3 - <<'PYX'
import pathlib
p=pathlib.Path("tools/solver_battery.py"); s=p.read_text()
s=s.replace("        return assert_(x * x * x >= 0.0)","        return assert_(x * x * x * x >= 0.0)",1)
p.write_text(s)
PYX
run "test_the_cheap_rows_answer_what_their_obligation_forces[6]"
rest

echo "--- C10: the no-backend run stops accounting for its cells"
snap
python3 - <<'PYX'
import pathlib
p=pathlib.Path("tools/solver_battery.py"); s=p.read_text()
s=s.replace('    if not groups:\n        return ""','    if True:\n        return ""',1)
p.write_text(s)
PYX
run test_it_runs_with_no_backend_and_names_every_cell_it_could_not_measure
rest

echo "--- C11: the module-scope jax import, in the lane that has jax"
snap
sed -i 's/^import argparse$/import argparse\nimport jax  # PLANTED/' tools/solver_battery.py
run test_the_module_imports_jax_nowhere_at_module_scope
rest

echo "--- clean tree again:"
COLUMNS=100 PYTHONPATH=$WT/src timeout 900 $PY -m pytest -q tests/test_solver_battery.py 2>&1 | tail -1
