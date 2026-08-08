#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
#
# One command for a local environment that can run tests/property/.
#
#     tools/property_venv.sh                    # jax 0.11.0, into ~/.cache
#     tools/property_venv.sh 0.10.2             # jax 0.10.2, into ~/.cache
#     tools/property_venv.sh 0.11.0 /path/to/v  # explicit target
#
# WHY THIS EXISTS AS A SCRIPT AND NOT A PARAGRAPH. The property suite needs
# hypothesis; the shared jax venvs on this machine are used by several agents at
# once and MUST NOT acquire it. A README sentence saying "don't install into
# those" is advice. This is a refusal: the target is checked against them by
# resolved path and the script exits non-zero rather than proceed.
#
# It creates a throwaway venv and installs exactly four things: jax + jaxlib at
# the requested series, numpy, pytest, and hypothesis at the version pinned in
# pyproject.toml's dev group. stelling itself is NOT installed — the suite is
# driven with PYTHONPATH=<tree>/src, so one venv can be pointed at any worktree.
# That is the same mechanism tools/property_check.py uses to run the properties
# against an arbitrary tree.

set -euo pipefail

JAX_VERSION="${1:-0.11.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default OUTSIDE the checkout. A venv inside the tree is a new root entry, and
# `tests/test_sdist_contents.py` requires every root entry to be a recorded
# decision — so a convenience script that dropped one there would make the
# suite red for anyone who ran it.
TARGET="${2:-${XDG_CACHE_HOME:-$HOME/.cache}/stelling-property/jax-${JAX_VERSION}}"

# ── THE REFUSAL ──────────────────────────────────────────────────────────────
#
# A TWO-NAME DENYLIST WAS THE WRONG SHAPE, and it was measured to be. The
# earlier version refused exactly `/home/nick/venvs/stelling-jax` and
# `/home/nick/venvs/stelling-jax010` by resolved path — correctly, and by
# resolved path, so `..`, a trailing slash and a symlink were all caught. What
# it let through, on this box, on the day it shipped:
#
#   /home/nick/venvs/stelling-jax/subdir   inside a venv it is told to protect
#   /home/nick/venvs/stelling-ci011        an existing venv, another agent's
#   /home/nick/venvs/jax051                an existing venv, another agent's
#   ~/.cache/stelling-property/jax-0.11.0  ITS OWN DEFAULT TARGET, which
#                                          already existed, and which `uv venv`
#                                          would have silently recreated
#
# The last one is the point. A denylist protects the venvs somebody thought to
# name; the thing that actually needs protecting is "a venv this script did not
# create", and only one of those is knowable by name.
#
# So there are three refusals, cheapest first, and the third is the general one:
#
#   1. the named shared venvs, kept because a named refusal is legible;
#   2. anything INSIDE one of them, by resolved-path prefix;
#   3. any existing directory that looks like a venv and does not carry this
#      script's own marker file, plus any existing non-empty directory that is
#      not a venv at all. Re-running this script on a target it made before is
#      fine and stays fine: that is what the marker is for.
FORBIDDEN=(
  "/home/nick/venvs/stelling-jax"
  "/home/nick/venvs/stelling-jax010"
)

# Written into every venv this script creates, and the only thing that makes a
# pre-existing directory reusable. Do not rename it without reading (3) above.
MARKER=".stelling-property-venv"

resolve() { python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

TARGET_ABS="$(resolve "$TARGET")"

refuse() {
  echo "REFUSED: $1" >&2
  echo "         target was: $TARGET_ABS" >&2
  echo "         Pass a different target, or delete it yourself if you are" >&2
  echo "         sure. This script will not remove a directory it did not" >&2
  echo "         create — several agents share this machine." >&2
  exit 2
}

# (1) and (2): the named shared venvs, and anything under them.
for bad in "${FORBIDDEN[@]}"; do
  [ -e "$bad" ] || continue
  bad_abs="$(resolve "$bad")"
  if [ "$TARGET_ABS" = "$bad_abs" ]; then
    refuse "$bad_abs is a shared venv."
  fi
  case "$TARGET_ABS" in
    "$bad_abs"/*)
      refuse "that path is INSIDE the shared venv $bad_abs."
      ;;
  esac
done

# (3) the general one: an existing directory this script did not make.
if [ -e "$TARGET_ABS" ]; then
  if [ ! -d "$TARGET_ABS" ]; then
    refuse "that path exists and is not a directory."
  elif [ -e "$TARGET_ABS/$MARKER" ]; then
    echo "== reusing a target this script created earlier (marker present)"
  elif [ -e "$TARGET_ABS/pyvenv.cfg" ] || [ -e "$TARGET_ABS/bin/python" ]; then
    refuse "that directory is already a venv, and it carries no $MARKER — so
         this script did not create it and something else may be using it.
         (This is what would have happened to the DEFAULT target: it already
         exists on this box and uv venv would have recreated it in silence.)"
  elif [ -n "$(ls -A "$TARGET_ABS" 2>/dev/null)" ]; then
    refuse "that directory exists and is not empty."
  fi
fi

# The hypothesis requirement is READ OUT OF pyproject.toml rather than repeated
# here. Two places to type a version is one place for them to disagree.
HYP_REQ="$(python3 - "$HERE/pyproject.toml" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'^dev\s*=\s*\[(.*?)\]', text, re.S | re.M)
if not m:
    sys.exit("no [dependency-groups] dev list in pyproject.toml")
reqs = re.findall(r'"([^"]+)"', m.group(1))
hyp = [r for r in reqs if re.match(r'^hypothesis\b', r)]
if not hyp:
    sys.exit("pyproject.toml's dev group does not pin hypothesis")
print(hyp[0])
PY
)"

echo "== target      : $TARGET_ABS"
echo "== jax series  : $JAX_VERSION"
echo "== hypothesis  : $HYP_REQ   (read from pyproject.toml)"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 "$TARGET_ABS"
  VIRTUAL_ENV="$TARGET_ABS" uv pip install \
    "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" "pytest>=8" "$HYP_REQ"
else
  python3 -m venv "$TARGET_ABS"
  "$TARGET_ABS/bin/python" -m pip install --upgrade pip
  "$TARGET_ABS/bin/python" -m pip install \
    "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" "pytest>=8" "$HYP_REQ"
fi

# Claim it, so a re-run of this script may reuse it and nothing else is
# mistaken for it. Written after the venv exists, so an interrupted create
# leaves an UNCLAIMED directory and the refusal above fires next time — which
# is the safe direction.
printf 'created by tools/property_venv.sh, jax %s\n' "$JAX_VERSION" \
  > "$TARGET_ABS/$MARKER"

echo
echo "== resolved"
"$TARGET_ABS/bin/python" - <<'PY'
import sys
import hypothesis, jax, numpy, pytest
print(f"   python     {sys.version.split()[0]}")
print(f"   jax        {jax.__version__}")
print(f"   numpy      {numpy.__version__}")
print(f"   hypothesis {hypothesis.__version__}")
print(f"   pytest     {pytest.__version__}")
PY

cat <<EOF

== run the property suite against THIS tree
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=${HERE}/src \\
     ${TARGET_ABS}/bin/python -m pytest -ra ${HERE}/tests/property

== run it against SOME OTHER worktree or revision
   ${TARGET_ABS}/bin/python ${HERE}/tools/property_check.py --tree /path/to/worktree
   ${TARGET_ABS}/bin/python ${HERE}/tools/property_check.py --rev  <sha-or-branch>

== demonstrate every positive control (each must FAIL where it is supposed to)
   ${TARGET_ABS}/bin/python ${HERE}/tools/property_check.py --controls
EOF
