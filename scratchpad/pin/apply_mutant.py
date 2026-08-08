# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The mutants for the shared-point pin repair, applied IN PLACE.

Run from the ROOT of a worktree that is nothing else:

    git worktree add --detach ../mut/M4_affine_and_private 0ad22bb
    cd ../mut/M4_affine_and_private
    python scratchpad/pin/apply_mutant.py M4_affine_and_private

`__pycache__` must be cleared and `python -B` used for every run in a
mutant tree: `.pyc` invalidation keys on `(mtime, size)`, so a same-size
edit inside one mtime tick collides and gives a confidently wrong
coverage answer. Measured in this project.

THE TWO FAMILIES, and why the first one is the finding.

* `M4` / `M5` / `M6` — a leg keeps a PRIVATE COPY of a shared decision and
  demotes the shared function to an `and`-veto. The call still happens,
  unconditionally, as the FIRST operand, with every real keyword
  argument, so an argument-recording pin sees exactly what it expects and
  a pin that forces the decision FALSE still sees the leg withhold. Only
  a pin that forces it TRUE can tell these apart from the real build.
  All three pass the whole suite at `0ad22bb`.
* `M2g` / `M3g` — a GENUINE inlining, the call gone and the expression
  written out. These are what `scratchpad/cert/apply_mutant.py`'s
  `M2_inline_setref_interval` / `M3_inline_setref_affine` are named for
  and are not: those keep the call and drop its third keyword argument.
  A genuine inlining reddens two tests per leg, not one.
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


_AFFINE_PRIVATE = '''def _affine_own_set_refutation(propagation) -> bool:
    """MUTANT: a private copy of the run-level rule; the shared call is
    demoted to an and-veto."""
    return propagation.region_inhabited or (
        (not propagation.narrowing_uncertified) and not propagation.assume_dropped
    )


_FR0 = Fraction(0)'''


def _affine_and_veto():
    p = pathlib.Path("src/stelling/affine.py")
    s = p.read_text()
    old = """            and not exactness.certifies_set_refutation(
                nonemptiness_certified=not propagation.narrowing_uncertified,
                assume_dropped=propagation.assume_dropped,"""
    new = """            and not (exactness.certifies_set_refutation(
                nonemptiness_certified=not propagation.narrowing_uncertified,
                assume_dropped=propagation.assume_dropped,"""
    assert old in s, "affine anchor missing"
    s = s.replace(old, new, 1)
    old2 = """                region_inhabited=propagation.region_inhabited,
            )
        ):"""
    new2 = """                region_inhabited=propagation.region_inhabited,
            ) and _affine_own_set_refutation(propagation))
        ):"""
    assert old2 in s, "affine tail anchor missing"
    s = s.replace(old2, new2, 1)
    assert "_FR0 = Fraction(0)" in s
    s = s.replace("_FR0 = Fraction(0)", _AFFINE_PRIVATE, 1)
    p.write_text(s)


@mutant("M4_affine_and_private")
def m4():
    """The AFFINE leg keeps a private copy and vetoes with the shared one.
    Must redden `test_both_legs_follow_the_shared_point_in_the_TRUE_direction`
    and NOTHING else."""
    _affine_and_veto()


@mutant("M5_both_and_private")
def m5():
    """Both legs do it."""
    _affine_and_veto()
    p = pathlib.Path("src/stelling/propagate.py")
    s = p.read_text()
    old = """    if exactness.certifies_set_refutation(
        nonemptiness_certified=not p.narrowing_uncertified,
        assume_dropped=p.assume_dropped,
        region_inhabited=p.region_inhabited,
    ):
        return
    mechanism = _uncertified_mechanism(p)"""
    new = """    if exactness.certifies_set_refutation(
        nonemptiness_certified=not p.narrowing_uncertified,
        assume_dropped=p.assume_dropped,
        region_inhabited=p.region_inhabited,
    ) and _interval_own_set_refutation(p):
        return
    mechanism = _uncertified_mechanism(p)"""
    assert old in s, "interval anchor missing"
    s = s.replace(old, new, 1)
    old2 = "def _uncertified_mechanism(p) -> str:"
    new2 = '''def _interval_own_set_refutation(p) -> bool:
    """MUTANT: a private copy of the run-level rule; the shared call is
    demoted to an and-veto."""
    return p.region_inhabited or (
        (not p.narrowing_uncertified) and not p.assume_dropped
    )


def _uncertified_mechanism(p) -> str:'''
    assert old2 in s
    p.write_text(s.replace(old2, new2, 1))


@mutant("M6_nonemptiness_and_private")
def m6():
    """The PER-VARIABLE decision, same trick. Written on ONE line so the
    file's line count is unchanged: `docs/supported-primitives.md` cites
    `propagate.py:LINE`, and a shifted citation is an artifact that would
    be read as a caught mutant."""
    p = pathlib.Path("src/stelling/propagate.py")
    s = p.read_text()
    old = """                exactness.certifies_nonemptiness(
                    self.exact, target_atom.id, definitely_true=def_true
                )
                and not ("""
    new = """                exactness.certifies_nonemptiness(
                    self.exact, target_atom.id, definitely_true=def_true
                ) and (def_true or target_atom.id in self.exact)
                and not ("""
    assert old in s, "M6 anchor missing"
    before = len(s.splitlines())
    s = s.replace(old, new, 1)
    assert len(s.splitlines()) == before, "M6 must not shift line numbers"
    p.write_text(s)


@mutant("M2g_genuine_inline_interval")
def m2g():
    """A GENUINE inlining on the interval leg: the shared call is gone."""
    p = pathlib.Path("src/stelling/propagate.py")
    s = p.read_text()
    old = """    if exactness.certifies_set_refutation(
        nonemptiness_certified=not p.narrowing_uncertified,
        assume_dropped=p.assume_dropped,
        region_inhabited=p.region_inhabited,
    ):
        return
    mechanism = _uncertified_mechanism(p)"""
    new = """    if p.region_inhabited or (
        (not p.narrowing_uncertified) and not p.assume_dropped
    ):
        return
    mechanism = _uncertified_mechanism(p)"""
    assert old in s, "M2g anchor missing"
    p.write_text(s.replace(old, new, 1))


@mutant("M3g_genuine_inline_affine")
def m3g():
    """A GENUINE inlining on the affine leg: the shared call is gone."""
    p = pathlib.Path("src/stelling/affine.py")
    s = p.read_text()
    start = s.index("            and not exactness.certifies_set_refutation(")
    end = s.index("        ):", start)
    new = """            and not (
                propagation.region_inhabited
                or (
                    (not propagation.narrowing_uncertified)
                    and not propagation.assume_dropped
                )
            )
"""
    p.write_text(s[:start] + new + s[end:])


if __name__ == "__main__":
    name = sys.argv[1]
    if name not in MUTANTS:
        raise SystemExit(f"unknown mutant {name!r}; have {sorted(MUTANTS)}")
    MUTANTS[name]()
    print(f"applied {name}")
