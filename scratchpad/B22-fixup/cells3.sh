#!/usr/bin/env bash
WT=/home/nick/MSF/stelling-wt/B22-fx-cells
for spec in "stelling-jax010 0" "stelling-jax010 1" "stelling-jax 1"; do
  set -- $spec; v=$1; x=$2
  find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  echo "=== FULL SUITE venv=$v x64=$x ==="
  (cd "$WT" && COLUMNS=110 JAX_PLATFORMS=cpu JAX_ENABLE_X64=$x \
     PYTHONPATH=$WT/src timeout 2400 /home/nick/venvs/$v/bin/python \
     -m pytest tests/ -q -p no:randomly 2>&1 | grep -E "^FAILED|^ERROR|passed|failed" | tail -6)
done
