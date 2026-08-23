#!/usr/bin/env bash
# One mutation at a time: apply, assert it applied, run the named gate, restore.
set -uo pipefail
WT=/home/nick/MSF/stelling-wt/B22-fx
PY=/home/nick/venvs/stelling-jax/bin/python
export COLUMNS=110 JAX_PLATFORMS=cpu PYTHONPATH=$WT/src

run() {  # run <label> <file-to-mutate> <python-mutation-snippet> <pytest -k expr> [extra files]
  local label="$1" target="$2" snippet="$3" kexpr="$4"
  find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  git -C "$WT" diff --quiet -- "$target" || { echo "$label: TREE DIRTY at $target -- aborting"; return 1; }
  "$PY" - "$WT/$target" <<PYEOF
import pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text(encoding="utf-8")
$snippet
p.write_text(t, encoding="utf-8")
PYEOF
  if git -C "$WT" diff --quiet -- "$target"; then
    echo "$label: MUTATION DID NOT APPLY -- result means nothing"
    return 1
  fi
  local out
  out=$(cd "$WT" && "$PY" -m pytest tests/ -q -k "$kexpr" -p no:randomly 2>&1 | tail -1)
  git -C "$WT" checkout -- "$target"
  find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  case "$out" in
    *failed*) echo "$label: REDDENS   -- $out" ;;
    *)        echo "$label: STAYED GREEN -- $out" ;;
  esac
}
