# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy <nicholas.roy@formulearn.org>
# SPDX-License-Identifier: Apache-2.0
"""Build the mutation worktrees for the shared-point pin audit, from scratch.

Every mutant gets its OWN worktree at ``0ad22bb``.  ``.pyc`` invalidation
keys on ``(mtime, size)``, so a same-size mutation applied inside one
mtime tick in a shared checkout can be masked by a stale cache and give a
confidently wrong coverage answer; a fresh tree per mutant plus ``-B``
removes that failure mode entirely.

Two families, and the contrast between them is the whole experiment:

``OLD/<name>``
    ``0ad22bb`` source + the mutation + the ``0ad22bb`` TESTS.
    Asks: *do the pins that were already in the tree see this?*

``NEW/<name>``
    the same source + the mutation + the WORKING TREE's tests.
    Asks: *does the pin I added see it?*

Usage::

    python scratchpad/pin/mutants.py build     # create every tree
    python scratchpad/pin/mutants.py list      # show applied diffs
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/nick/MSF/stelling")
ROOT = Path("/home/nick/MSF/.wt-pin/mine")
W = Path("/home/nick/MSF/.wt-pin/W")
BASE = "0ad22bb"

# tests carried from the working tree into the NEW family
CARRY = ("tests/test_exactness_lift.py", "tests/test_nonempty_certificate.py")


# --- the mutations ------------------------------------------------------------
#
# Each is (file, old_exact_text, new_text).  Applied by exact replacement
# and asserted to have changed the file, so a mutation that silently
# failed to apply cannot be mistaken for a mutation the pins missed.

_AFFINE_PRIVATE = '''
def _affine_own_set_refutation(propagation) -> bool:
    """MUTANT: a private copy of the run-level rule.  The shared call is
    kept -- unconditionally, first operand, all three real kwargs -- and
    demoted to an and-veto."""
    return propagation.region_inhabited or (
        (not propagation.narrowing_uncertified)
        and not propagation.assume_dropped
    )


_FR0 = Fraction(0)
'''

_INTERVAL_PRIVATE = '''
def _interval_own_set_refutation(p) -> bool:
    """MUTANT: a private copy of the run-level rule.  The shared call is
    kept -- unconditionally, first operand, all three real kwargs -- and
    demoted to an and-veto."""
    return p.region_inhabited or (
        (not p.narrowing_uncertified) and not p.assume_dropped
    )


def _uncertified_mechanism(p) -> str:
'''

M_AFFINE_AND = [
    ("src/stelling/affine.py", "\n_FR0 = Fraction(0)\n", _AFFINE_PRIVATE),
    (
        "src/stelling/affine.py",
        "            and not exactness.certifies_set_refutation(",
        "            and not (exactness.certifies_set_refutation(",
    ),
    (
        "src/stelling/affine.py",
        "                region_inhabited=propagation.region_inhabited,\n            )\n        ):",
        "                region_inhabited=propagation.region_inhabited,\n            ) and _affine_own_set_refutation(propagation))\n        ):",
    ),
]

M_INTERVAL_AND = [
    (
        "src/stelling/propagate.py",
        "\ndef _uncertified_mechanism(p) -> str:\n",
        _INTERVAL_PRIVATE,
    ),
    (
        "src/stelling/propagate.py",
        "        region_inhabited=p.region_inhabited,\n    ):\n        return\n    mechanism = _uncertified_mechanism(p)",
        "        region_inhabited=p.region_inhabited,\n    ) and _interval_own_set_refutation(p):\n        return\n    mechanism = _uncertified_mechanism(p)",
    ),
]

M_NONEMPTINESS_AND = [
    (
        "src/stelling/propagate.py",
        "                exactness.certifies_nonemptiness(\n"
        "                    self.exact, target_atom.id, definitely_true=def_true\n"
        "                )\n",
        "                exactness.certifies_nonemptiness(\n"
        "                    self.exact, target_atom.id, definitely_true=def_true\n"
        "                ) and (def_true or target_atom.id in self.exact)\n",
    ),
]

# A GENUINE inlining: the call is GONE and the rule is written out in
# place.  Contrast with the certificate branch's `M2_inline_setref_*`,
# which keep the call and only drop the third keyword argument.
M_INLINE_INTERVAL = [
    (
        "src/stelling/propagate.py",
        "    if exactness.certifies_set_refutation(\n"
        "        nonemptiness_certified=not p.narrowing_uncertified,\n"
        "        assume_dropped=p.assume_dropped,\n"
        "        region_inhabited=p.region_inhabited,\n"
        "    ):",
        "    if p.region_inhabited or (\n"
        "        (not p.narrowing_uncertified) and not p.assume_dropped\n"
        "    ):",
    ),
]

_AFFINE_CALL_BLOCK = """            and not exactness.certifies_set_refutation(
                nonemptiness_certified=not propagation.narrowing_uncertified,
                assume_dropped=propagation.assume_dropped,
                # THE THIRD INPUT: the non-emptiness CERTIFICATE the
                # interval leg computed, carried on the propagation. It is
                # passed here for exactly the reason the other two are —
                # the run's WHOLE assume state, through the one shared
                # function. A leg that read the certificate and decided
                # for itself would restore the coincidence this call
                # exists to remove, one argument later.
                region_inhabited=propagation.region_inhabited,
            )
"""

M_INLINE_AFFINE = [
    (
        "src/stelling/affine.py",
        _AFFINE_CALL_BLOCK,
        "            and not (  # MUTANT: the call is GONE, rule written out\n"
        "                propagation.region_inhabited\n"
        "                or (\n"
        "                    (not propagation.narrowing_uncertified)\n"
        "                    and not propagation.assume_dropped\n"
        "                )\n"
        "            )\n",
    ),
]

# A GRANTED answer treated as two-sided: the leg reads a True as licence
# to DECIDE rather than merely to stop withholding.  Scoped to runs that
# would otherwise have withheld, which is what keeps it out of the way of
# every ordinary run and makes it the shape a False-only pin cannot see.
M_TWO_SIDED_GRANT = [
    (
        "src/stelling/propagate.py",
        "        region_inhabited=p.region_inhabited,\n    ):\n        return\n",
        "        region_inhabited=p.region_inhabited,\n"
        "    ):\n"
        "        # MUTANT: a granted answer used to DECIDE, not only to\n"
        "        # stop withholding.\n"
        "        if p.narrowing_uncertified or p.assume_dropped:\n"
        "            for _i, _o in enumerate(p.obligations):\n"
        "                if _o.status == 'unknown':\n"
        "                    p.obligations[_i] = dataclasses.replace(\n"
        "                        _o, status='discharged', detail='MUTANT'\n"
        "                    )\n"
        "        return\n",
    ),
]

MUTANTS = {
    "M4_affine_and_private": M_AFFINE_AND,
    "M7_granted_answer_decides": M_TWO_SIDED_GRANT,
    "M5_both_and_private": M_AFFINE_AND + M_INTERVAL_AND,
    "M6_nonemptiness_and_private": M_NONEMPTINESS_AND,
    "M2g_genuine_inline_interval": M_INLINE_INTERVAL,
    "M3g_genuine_inline_affine": M_INLINE_AFFINE,
    "CTRL_unmutated": [],
}


def build_one(family: str, name: str) -> Path:
    d = ROOT / family / name
    if d.exists():
        subprocess.run(
            ["git", "-C", str(REPO), "worktree", "remove", "--force", str(d)],
            capture_output=True,
        )
        shutil.rmtree(d, ignore_errors=True)
    d.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "add", "--detach", "-f",
         str(d), BASE],
        check=True,
        capture_output=True,
    )
    if family == "NEW":
        for rel in CARRY:
            shutil.copy(W / rel, d / rel)
    for rel, old, new in MUTANTS[name]:
        path = d / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(
                f"{family}/{name}: anchor occurs {text.count(old)}x in {rel}"
            )
        path.write_text(text.replace(old, new))
    # a mutation that did not land must never look like a pin that missed
    changed = subprocess.run(
        ["git", "-C", str(d), "status", "--short", "--", "src/"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if MUTANTS[name] and not changed:
        raise SystemExit(f"{family}/{name}: mutation did not land")
    if not MUTANTS[name] and changed:
        raise SystemExit(f"{family}/{name}: control tree is not clean")
    # it must still import
    subprocess.run(
        [sys.executable, "-B", "-c", "import ast,sys;"
         "[ast.parse(open(p).read()) for p in sys.argv[1:]]",
         str(d / "src/stelling/affine.py"),
         str(d / "src/stelling/propagate.py")],
        check=True,
    )
    return d


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "build"
    if what == "build":
        for family in ("OLD", "NEW"):
            for name in MUTANTS:
                print(family, name, build_one(family, name))
    elif what == "list":
        for family in ("OLD", "NEW"):
            for name in MUTANTS:
                d = ROOT / family / name
                print(f"===== {family}/{name} =====")
                print(
                    subprocess.run(
                        ["git", "-C", str(d), "diff", "--stat", "--", "src/"],
                        capture_output=True,
                        text=True,
                    ).stdout
                )


if __name__ == "__main__":
    main()
