#!/usr/bin/env bash
# The FOUR-CELL full-suite drive: {jax 0.10.2, jax 0.11.0} x {x64 off, x64 on}.
#
#   ./cells3.sh [<worktree>]        # default: the worktree this script is in
#
# WHAT THIS FILE GOT WRONG THE FIRST TIME, and why each line below is here.
# As committed at 725edc3 it was unusable as an artefact in four ways:
#
#   1. `WT=` was hard-coded to a worktree that no longer existed, so every
#      `cd "$WT"` failed and it ran pytest ZERO times while printing three
#      apparently-clean headings. An instrument reporting a clean result
#      because it did not run -- the thing NORM_mutation_and_worktrees.md is
#      about -- in the harness written to check for exactly that.
#   2. It drove three cells and called itself the four-cell script: the
#      `stelling-jax x64=0` cell was missing.
#   3. It never checked `stelling.__file__`, so a PYTHONPATH slip would have
#      silently tested the MAIN checkout from inside a worktree (norm 2).
#   4. It piped pytest into `grep` and reported grep's exit status, so a
#      collection crash rendered as a blank line rather than as a failure.
set -uo pipefail

WT=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
[ -d "$WT/src/stelling" ] || { echo "not a stelling worktree: $WT" >&2; exit 2; }
echo "worktree: $WT  ($(git -C "$WT" rev-parse --short HEAD))"

rc=0
for spec in "stelling-jax010 0" "stelling-jax010 1" "stelling-jax 0" "stelling-jax 1"; do
  set -- $spec; venv=$1; x=$2
  PY=/home/nick/venvs/$venv/bin/python
  [ -x "$PY" ] || { echo "=== $venv x64=$x: NO SUCH VENV -- NOT RUN"; rc=1; continue; }

  find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null

  # NORM 2: prove the import resolves into THIS worktree before believing a
  # single figure from this cell.
  resolved=$(cd "$WT" && JAX_PLATFORMS=cpu PYTHONPATH="$WT/src" "$PY" \
             -c "import stelling; print(stelling.__file__)" 2>&1)
  case "$resolved" in
    "$WT"/src/stelling/*) ;;
    *) echo "=== $venv x64=$x: stelling resolves to $resolved -- NOT RUN"; rc=1; continue ;;
  esac
  ver=$(cd "$WT" && JAX_PLATFORMS=cpu "$PY" -c "import jax;print(jax.__version__)" 2>/dev/null)

  echo "=== FULL SUITE venv=$venv (jax $ver) x64=$x ==="
  out=$(cd "$WT" && COLUMNS=110 JAX_PLATFORMS=cpu JAX_ENABLE_X64=$x \
        PYTHONPATH="$WT/src" timeout 3600 "$PY" -m pytest tests/ -q -p no:randomly 2>&1)
  status=$?                       # pytest's OWN status, not a grep's
  printf '%s\n' "$out" | grep -E "^FAILED|^ERROR|passed|failed|error" | tail -6
  echo "    exit status: $status"
  [ "$status" -eq 0 ] || rc=1
done
exit $rc
