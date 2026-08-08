# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Two invariants the certificate's soundness argument rests on, measured.

**I1 — a certifying probe narrows NOTHING.** The witness answer for each
assume is read BEFORE `_assume_constrain` can meet anything into the env,
and the argument that this is enough is: a predicate whose box is `[1, 1]`
is definitely true over the boxes in force, so the meet with the closed
half-space is a no-op. If that failed, an earlier assume could certify
itself AND cut the box a later assume is read against — and the later
`[1, 1]` would be a statement about a box that no longer over-approximates
the point. Measured here rather than argued: on every probe run that
certifies, no note says a var was NARROWED.

**I2 — ieee.** Under `semantics="ieee"` a pinned declaration is
subnormal-HAZED, so the "point" box is a hull rather than a point. A
definite TRUE over a hull is true under both flush semantics, so the
witness is still sound — and a maybe-NaN bool is `[0, 1]`, never `[1, 1]`,
so the channel closes itself on flagged predicates.

**MEASURED, and it corrected the sentence that stood here.** This
docstring claimed "ieee is the more conservative one wherever they
differ". It is not, and `r20_sqrt_no_slack` is the counterexample: `x` is
declared at the point 0.25, and `sqrt(0.25)` is EXACTLY 0.5 in binary64,
so the ieee transfer encloses it as the point `[0.5, 0.5]` and
`>= 0.5` is definitely TRUE — while the REAL-mode transfer bumps outward
unconditionally, straddles the bound, and certifies nothing. ieee
certifies there and real does not. Both are sound **for their own dial**:
the certificate is computed in the same semantics the query is judged in,
and under ieee the program really does compute 0.5. Three other rows
(`r15`, `r17`, `r18` — the int32 and float32 declarations) go the other
way, ieee declining where real certifies. The honest statement is that
the two dials are not ordered, only each internally consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import corpus as C  # noqa: E402
from ledger import _harness  # noqa: E402
from stelling import exactness, propagate as P  # noqa: E402
from stelling.harness import trace  # noqa: E402


def main():
    print("=== I1: a certifying probe narrows nothing ===")
    bad = 0
    checked = 0
    for row in C.ROWS:
        closed = trace(_harness(row))
        required = P._assume_equation_ids(closed.jaxpr)
        if not required:
            continue
        for k in range(P._PROBE_COUNT):
            probe = P._Propagator("constrain", "real")
            probe.pin = k
            try:
                probe.run(closed.jaxpr, list(closed.consts), [])
            except Exception:  # noqa: BLE001
                continue
            if not exactness.certifies_point_witness(
                required_assumes=required,
                witnessed_assumes=frozenset(
                    key for key, ok in probe.assume_witness.items() if ok
                ),
            ):
                continue
            checked += 1
            narrowing = [n for n in probe.notes if "narrowed var" in n]
            if narrowing or probe.narrowing_uncertified:
                bad += 1
                print(f"  !! {row.__name__} probe {k}: {narrowing}")
    print(f"  certifying probe runs inspected: {checked}; "
          f"runs that narrowed anything: {bad}")

    print()
    print("=== I2: real vs ieee, same rows ===")
    disagree = 0
    for row in C.ROWS:
        closed = trace(_harness(row))
        try:
            r = P.propagate(closed, semantics="real")
            i = P.propagate(closed, semantics="ieee")
        except Exception as e:  # noqa: BLE001
            print(f"  {row.__name__:38s} raised {type(e).__name__}")
            continue
        mark = "" if r.region_inhabited == i.region_inhabited else "  <-- differs"
        if mark:
            disagree += 1
        print(f"  {row.__name__:38s} real={str(r.region_inhabited):5s} "
              f"ieee={str(i.region_inhabited):5s}{mark}")
    print(f"  rows whose certificate differs by dial: {disagree} "
          f"(NOT ordered — see this file's docstring: real is more "
          f"conservative on r20, ieee on r15/r17/r18)")


if __name__ == "__main__":
    main()
