#!/usr/bin/env bash
# One mutation at a time: apply, ASSERT IT APPLIED, run the named gate, restore.
#
#   source mutate.sh
#   run <label> <file(s)> <python-snippet on `t`> <pytest paths> [venv] [x64]
#
# The snippet edits the string `t`; a mutation that leaves the file unchanged
# is reported as meaning nothing rather than as a result.
set -uo pipefail
WT=${WT:-/home/nick/MSF/stelling-wt/B22-fx2-mut}

run() {
  local label="$1" target="$2" snippet="$3" paths="$4"
  local venv="${5:-stelling-jax}" x64="${6:-0}"
  local PY=/home/nick/venvs/$venv/bin/python
  find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  git -C "$WT" diff --quiet || { echo "$label: TREE DIRTY -- aborting"; return 1; }
  local f
  for f in $target; do
    "$PY" - "$WT/$f" <<PYEOF
import pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text(encoding="utf-8")
$snippet
p.write_text(t, encoding="utf-8")
PYEOF
  done
  if git -C "$WT" diff --quiet; then
    echo "$label: MUTATION DID NOT APPLY -- result means nothing"
    return 1
  fi
  local applied; applied=$(git -C "$WT" diff --stat | tail -1)
  local out
  out=$(cd "$WT" && COLUMNS=110 JAX_PLATFORMS=cpu JAX_ENABLE_X64=$x64 \
        PYTHONPATH="$WT/src" "$PY" -m pytest $paths -q -p no:randomly 2>&1 | tail -1)
  git -C "$WT" checkout -- .
  find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  case "$out" in
    *failed*|*error*) echo "$label [$venv x64=$x64]: REDDENS       -- $out" ;;
    *)                echo "$label [$venv x64=$x64]: STAYED GREEN  -- $out   ($applied)" ;;
  esac
}
