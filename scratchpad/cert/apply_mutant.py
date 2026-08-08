# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Apply one BEHAVIOUR-PRESERVING routing mutant to a worktree in place.

Each mutant computes exactly what the shipped code computes and reaches it
by a different route — so no verdict in the tree moves and only a ROUTING
pin can see it. Run inside the mutant's own worktree:

    python scratchpad/cert/apply_mutant.py M1_inline_witness

`__pycache__` must be cleared and `python -B` used for every run in the
mutant tree: `.pyc` invalidation keys on `(mtime, size)`, so a same-size
edit inside one mtime tick collides and gives a confidently wrong coverage
answer. Measured in this project.
"""

from __future__ import annotations

import pathlib
import sys

MUTANTS = {}


def mutant(name):
    def deco(fn):
        MUTANTS[name] = fn
        return fn

    return deco


@mutant("M1_inline_witness")
def m1():
    """Inline the WITNESS decision in propagate.py: identical value,
    private copy. Must redden `test_the_witness_route_is_the_shared_primitive_too`."""
    p = pathlib.Path("src/stelling/propagate.py")
    s = p.read_text()
    old = """        if exactness.certifies_point_witness(
            required_assumes=required,
            witnessed_assumes=frozenset(
                key for key, ok in probe.assume_witness.items() if ok
            ),
        ):"""
    new = """        witnessed = frozenset(
            key for key, ok in probe.assume_witness.items() if ok
        )
        if bool(required) and required <= witnessed:"""
    assert old in s, "M1 anchor missing"
    p.write_text(s.replace(old, new))


@mutant("M2_inline_setref_interval")
def m2():
    """The INTERVAL leg lifts the withholding locally instead of passing
    the certificate into the shared decision. Must redden
    `test_every_reach_of_the_shared_point_names_the_certificate`."""
    p = pathlib.Path("src/stelling/propagate.py")
    s = p.read_text()
    old = """    if exactness.certifies_set_refutation(
        nonemptiness_certified=not p.narrowing_uncertified,
        assume_dropped=p.assume_dropped,
        region_inhabited=p.region_inhabited,
    ):
        return
    mechanism = _uncertified_mechanism(p)"""
    new = """    if p.region_inhabited or exactness.certifies_set_refutation(
        nonemptiness_certified=not p.narrowing_uncertified,
        assume_dropped=p.assume_dropped,
    ):
        return
    mechanism = _uncertified_mechanism(p)"""
    assert old in s, "M2 anchor missing"
    p.write_text(s.replace(old, new))


@mutant("M3_inline_setref_affine")
def m3():
    """The AFFINE leg does the same. Must redden the same pin."""
    p = pathlib.Path("src/stelling/affine.py")
    s = p.read_text()
    old = "                region_inhabited=propagation.region_inhabited,\n            )\n        ):"
    new = "            )\n            and not propagation.region_inhabited\n        ):"
    assert old in s, "M3 anchor missing"
    p.write_text(s.replace(old, new))


if __name__ == "__main__":
    name = sys.argv[1]
    MUTANTS[name]()
    print(f"{name} applied")
