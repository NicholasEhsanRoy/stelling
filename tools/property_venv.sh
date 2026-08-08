#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
#
# One command for a local environment that can run tests/property/.
#
#     tools/property_venv.sh                    # jax 0.11.0, ./.venv-prop-0.11.0
#     tools/property_venv.sh 0.10.2             # jax 0.10.2, ./.venv-prop-0.10.2
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
TARGET="${2:-${HERE}/.venv-prop-${JAX_VERSION}}"

# The two shared venvs, named so the refusal is legible rather than implicit.
FORBIDDEN=(
  "/home/nick/venvs/stelling-jax"
  "/home/nick/venvs/stelling-jax010"
)

resolve() { python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

TARGET_ABS="$(resolve "$TARGET")"
for bad in "${FORBIDDEN[@]}"; do
  if [ -e "$bad" ] && [ "$TARGET_ABS" = "$(resolve "$bad")" ]; then
    echo "REFUSED: $TARGET_ABS is a shared venv. Pick another target." >&2
    exit 2
  fi
done

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
