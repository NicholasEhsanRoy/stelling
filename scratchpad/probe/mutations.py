# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The mutation survey behind clause C10 of `scratchpad/PREREG_PROBE.md`.

Each entry is a one-line edit to `src/stelling/propagate.py` that removes a
behaviour the tests are supposed to hold. A mutation that leaves the suite
green is a behaviour nothing pins.

**Build each mutant in its OWN worktree, run with `python -B`, and clear
`__pycache__` first.** `.pyc` invalidation keys on `(mtime, size)`, so two
same-size mutations written inside one mtime tick collide and the survey
reports a confidently wrong answer. Measured; do not edit in place.

    python scratchpad/probe/mutations.py build <sha> <dir>
    python scratchpad/probe/mutations.py run <dir>

Result on `fix/probe-membership` (jax 0.11.0, `tests/test_probe_witness.py`
+ `tests/test_branch_reachability.py`, 55 tests): every mutation below
fails at least one test; the unmutated control passes all 55.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = "/home/nick/MSF/stelling"
FILES = "tests/test_probe_witness.py tests/test_branch_reachability.py"

MUT = [
    # the integer rounding itself
    (
        "M6_no_integer_rounding",
        '        if _is_integer_dtype(dtype) or dtype == "bool":\n'
        "            v = float(math.floor(v + 0.5))",
        "        if False:\n            v = float(math.floor(v + 0.5))",
    ),
    # the defect this branch fixes: clamp to the raw box after rounding
    (
        "M6b_clamp_to_raw_box",
        "        v = min(max(v, m_lo), m_hi)",
        "        v = min(max(v, lo), hi)",
    ),
    # the second bound on the declared set
    ("M6c_no_dtype_range_clamp", "    if d_lo is not None:", "    if False:"),
    # the witness search's gate on a narrowing precondition
    (
        "M8_no_assume_gate",
        "    if p.any_constrained or p.assume_dropped:",
        "    if False:",
    ),
    # the grid's width
    ("M9_probe_count_3", "_PROBE_COUNT = 16", "_PROBE_COUNT = 3"),
    # nonvacuity's presence among the unexaminable obligations
    (
        "M12_assert_only",
        '_UNEXAMINABLE_OBLIGATIONS = frozenset({"stelling_assert", '
        '"stelling_nonvacuity"})',
        '_UNEXAMINABLE_OBLIGATIONS = frozenset({"stelling_assert"})',
    ),
    # the branch index in the reachability path
    (
        "M18_no_branch_index",
        "                    self._branch_path.append((id(eqn), i))",
        "                    self._branch_path.append((id(eqn), 0))",
    ),
    # the path in the reachability key
    (
        "N23_id_only_key",
        "        return (tuple(self._branch_path), id(eqn))",
        "        return (id(eqn),)",
    ),
    # the overflowing probe grid
    (
        "F4_overflow_grid",
        "            v = lo * (1.0 - f) + hi * f",
        "            v = lo + f * (hi - lo)",
    ),
    # the innermost swallowing primitive
    (
        "F5_outermost_swallower",
        "                    self._record_unexamined(e, swallower)",
        "                    self._record_unexamined(e, eqn.primitive)",
    ),
]


def build(sha, root):
    os.makedirs(root, exist_ok=True)
    for name, old, new in MUT:
        wt = os.path.join(root, name)
        if not os.path.isdir(wt):
            subprocess.run(
                ["git", "-C", REPO, "worktree", "add", "--detach", wt, sha],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        p = os.path.join(wt, "src/stelling/propagate.py")
        s = open(p).read()
        assert s.count(old) == 1, (name, s.count(old))
        open(p, "w").write(s.replace(old, new))
        print("built", name, flush=True)


def run(root, python="/home/nick/venvs/stelling-jax/bin/python"):
    for name, _old, _new in MUT:
        wt = os.path.join(root, name)
        subprocess.run(
            f"find {wt} -name __pycache__ -type d -prune -exec rm -rf {{}} + "
            f"2>/dev/null; cd {wt} && "
            f"{python} -B -c "
            f"\"import stelling,sys;assert '{name}' in stelling.__file__\" && "
            f"{python} -B -m pytest -q -p no:cacheprovider {FILES} 2>&1 "
            f"| tail -1 | sed 's/^/[{name}] /'",
            shell=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": f"{wt}/src"},
        )


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        raise SystemExit(__doc__)
