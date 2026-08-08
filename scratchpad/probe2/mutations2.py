# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The mutation recipe for the FLOAT half of the member gap.

Seven one-line edits to `src/stelling/propagate.py`, each the plausible
weakening of one surface the repair adds. A surface no mutation reddens
is a surface nothing holds.

Each mutation goes in its OWN worktree, is run with `python -B`, and has
`__pycache__` cleared first: `.pyc` invalidation keys on `(mtime, size)`,
so two same-size mutations made inside one mtime tick collide and the
survey answers confidently wrong. Measured in this project; do not edit
in place.

Outcomes, `tests/test_probe_witness.py`, jax 0.11.0, control 52 passed:
all seven redden. An eighth ran first and did NOT — deleting the
`max(m_lo, -big), min(m_hi, big)` clamp from the float path moved no
test, and it moved no obligation row in the 1359-row corpus either. That
was the clamp being dead rather than a surface being unheld: the directed
rounding already returns a value of the format or the ±inf that says the
format has nothing on that side. The clamp was removed instead of pinned,
and `E_float_box_never_empty` now mutates the emptiness check that really
does carry the weight.

    python scratchpad/probe2/mutations2.py apply <NAME> <WORKTREE>
    python scratchpad/probe2/mutations2.py names
"""
from __future__ import annotations

import os
import sys

# name -> (old, new). Each `old` must appear exactly once.
MUTATIONS = {
    # the directed rounding becomes round-to-nearest: nearest can cross
    # the endpoint it is narrowing, which is the whole reason the
    # direction is carried
    "A_nearest_not_directed": (
        "    s = math.ceil(scaled) if direction > 0 else math.floor(scaled)",
        "    s = round(scaled)",
    ),
    # the direction argument is ignored entirely
    "B_always_down": (
        "    s = math.ceil(scaled) if direction > 0 else math.floor(scaled)",
        "    s = math.floor(scaled)",
    ),
    # the float path of _member_bounds reverts to returning the raw box
    "C_float_bounds_inert": (
        "    fmt = _FLOAT_FORMATS.get(dtype)\n    if fmt is None:\n        return None, None\n    big = _FLOAT_MAX[dtype]",
        "    fmt = _FLOAT_FORMATS.get(dtype)\n    if fmt is None:\n        return None, None\n    return lo, hi\n    big = _FLOAT_MAX[dtype]",
    ),
    # the endpoints are narrowed but the interior points are not rounded:
    # the clamp alone leaves every interior probe on binary64's grid
    "D_no_value_rounding": (
        "        if fmt is not None:\n            # The clamp alone is not enough",
        "        if False:\n            # The clamp alone is not enough",
    ),
    # a box holding no value of the format stops being empty: the
    # endpoints cross and the probe is formed anyway, off the box
    "E_float_box_never_empty": (
        "    # A clamp did stand here; the mutation survey deleted it and no test\n"
        "    # moved, and it was removed rather than left looking load-bearing.\n"
        "    if m_lo > m_hi:\n        return None, None\n    return m_lo, m_hi",
        "    # A clamp did stand here; the mutation survey deleted it and no test\n"
        "    # moved, and it was removed rather than left looking load-bearing.\n"
        "    return m_lo, m_hi",
    ),
    # default-deny for a float format the table does not name becomes
    # "skip the clamp", which is what returned a non-member witness
    "F_unknown_float_passes": (
        "    fmt = _FLOAT_FORMATS.get(dtype)\n    if fmt is None:\n        return None, None",
        "    fmt = _FLOAT_FORMATS.get(dtype)\n    if fmt is None:\n        return lo, hi",
    ),
    # default-deny for an integer dtype the table does not name reverts
    # to skipping the clamp: this is int2 / uint2
    "G_unknown_int_passes": (
        "        d_lo, d_hi = _INT_DTYPE_BOUNDS.get(dtype, (None, None))\n        if d_lo is None:\n            return None, None",
        "        d_lo, d_hi = _INT_DTYPE_BOUNDS.get(dtype, (None, None))\n        if d_lo is None:\n            return lo, hi",
    ),
}


def apply(name, worktree):
    old, new = MUTATIONS[name]
    path = os.path.join(worktree, "src", "stelling", "propagate.py")
    s = open(path).read()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{name}: anchor appears {n} times, expected 1")
    open(path, "w").write(s.replace(old, new, 1))
    print(f"{name}: applied to {path}")


if __name__ == "__main__":
    if sys.argv[1] == "names":
        print("\n".join(sorted(MUTATIONS)))
    else:
        apply(sys.argv[2], sys.argv[3])
