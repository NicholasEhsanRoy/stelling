# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The per-obligation ledger: what the certificate moved, and whether the
oracle agrees that each move was sound.

SCORED PER OBLIGATION, NEVER PER QUERY. A corpus in this project once
scored per query and turned a measured 24:168 trade into a fake 216:216.

Run:  python scratchpad/cert/ledger.py [--mutant NAME]

``--mutant`` drives one of the POSITIVE CONTROLS: a deliberately unsound
variant of this build's own machinery that CAN reach the outcome the
ledger reports zero of. A zero with no positive control is unfalsifiable.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import corpus as C  # noqa: E402
import oracle as O  # noqa: E402
from stelling import exactness, propagate as P  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402


class _Stelling:
    """The backend a corpus row runs against in stelling mode."""

    np = jnp

    def any(self, shape, dtype, bounds):
        return any_array(shape, dtype, bounds)

    def assume(self, pred):
        assume(pred)

    def assert_(self, pred):
        self._obs.append(assert_(pred))

    def __init__(self):
        self._obs = []


def _harness(row):
    def h():
        b = _Stelling()
        row(b)
        return tuple(b._obs)

    return h


# --- the positive controls ----------------------------------------------------


def _mutant_two_sided(monkey):
    """UNSOUND: let the certificate lift a withheld obligation all the way
    to `discharged`. The ledger must report a move toward VERIFIED."""
    real = P._withhold_uncertified_refutations

    def two_sided(p):
        if p.region_inhabited:
            for i, o in enumerate(p.obligations):
                if o.status == "violated-over-set":
                    p.obligations[i] = dataclasses.replace(
                        o, status="discharged", detail="MUTANT two-sided"
                    )
        return real(p)

    monkey["_withhold_uncertified_refutations"] = P._withhold_uncertified_refutations
    P._withhold_uncertified_refutations = two_sided


def _mutant_certify_everything(monkey):
    """UNSOUND: certify every run inhabited, whatever the probe found. The
    oracle must flag the EMPTY-region rows as wrong refutations."""
    monkey["certifies_point_witness"] = exactness.certifies_point_witness
    exactness.certifies_point_witness = lambda **k: True


MUTANTS = {
    "two_sided": _mutant_two_sided,
    "certify_everything": _mutant_certify_everything,
}


def _restore(monkey):
    for name, fn in monkey.items():
        if name == "certifies_point_witness":
            exactness.certifies_point_witness = fn
        else:
            setattr(P, name, fn)


# --- the runs -----------------------------------------------------------------


def _statuses(closed, *, certificate):
    if certificate:
        return P.propagate(closed)
    saved = exactness.certifies_point_witness
    exactness.certifies_point_witness = lambda **k: False
    try:
        return P.propagate(closed)
    finally:
        exactness.certifies_point_witness = saved


def _verdict(h, *, certificate, refine=None):
    if certificate:
        return check(h, vacuity_mode="inputs-only", refine=refine).status
    saved = exactness.certifies_point_witness
    exactness.certifies_point_witness = lambda **k: False
    try:
        return check(h, vacuity_mode="inputs-only", refine=refine).status
    finally:
        exactness.certifies_point_witness = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", choices=sorted(MUTANTS), default=None)
    ap.add_argument("--samples", type=int, default=20000)
    args = ap.parse_args()

    monkey: dict = {}
    if args.mutant:
        MUTANTS[args.mutant](monkey)

    rows = []
    tot = {
        "obligations": 0,
        "recovered": 0,          # unknown -> violated-over-set
        "toward_discharged": 0,  # ANY move whose destination is discharged
        "off_discharged": 0,     # any obligation that LEFT discharged
        "other_move": 0,
        # a recovery the oracle can falsify outright: an ADMISSIBLE point
        # at which the obligation is TRUE, so REFUTED is definitely wrong
        "wrong_refuted": 0,
        # a recovery on a row where the oracle found NO admissible point
        # at all — the region may be empty and the refutation vacuous.
        # THIS is the metric that catches a certificate that lies, and the
        # one `wrong_refuted` structurally cannot: an empty region has no
        # admissible point to satisfy anything with.
        "vacuous_recovery": 0,
        "wrong_verified": 0,
        "label_disagreements": 0,
    }
    for row in C.ROWS:
        name = row.__name__
        h = _harness(row)
        closed = trace(h)
        try:
            before = _statuses(closed, certificate=False)
            after = _statuses(closed, certificate=True)
        except Exception as e:  # noqa: BLE001 — a raising row is a finding
            print(f"!! {name}: {type(e).__name__}: {str(e)[:110]}")
            continue
        nonempty, sat, viol, n_ob, n_adm = O.measure(row, samples=args.samples)
        labelled = name in C.LABELLED_INHABITED
        if labelled != nonempty:
            tot["label_disagreements"] += 1
        v_before = _verdict(h, certificate=False)
        v_after = _verdict(h, certificate=True)
        tot["obligations"] += len(after.obligations)
        moves = []
        for i, (b, a) in enumerate(zip(before.obligations, after.obligations)):
            if b.status == a.status:
                continue
            moves.append((i, b.status, a.status))
            if a.status == "discharged":
                tot["toward_discharged"] += 1
            elif b.status == "discharged":
                tot["off_discharged"] += 1
            elif (b.status, a.status) == ("unknown", "violated-over-set"):
                tot["recovered"] += 1
                # the oracle's two vetoes on THIS recovery
                if i < len(sat) and sat[i]:
                    tot["wrong_refuted"] += 1
                if not nonempty:
                    tot["vacuous_recovery"] += 1
            else:
                tot["other_move"] += 1
        # a VERIFIED the oracle can falsify, whatever produced it
        for i, a in enumerate(after.obligations):
            if a.status == "discharged" and i < len(viol) and viol[i]:
                tot["wrong_verified"] += 1
        rows.append(
            dict(
                name=name,
                label="INHABITED" if labelled else "EMPTY",
                oracle_nonempty=nonempty,
                admissible=n_adm,
                before=[o.status for o in before.obligations],
                after=[o.status for o in after.obligations],
                moves=moves,
                verdict=(v_before, v_after),
                inhabited_flag=after.region_inhabited,
                dropped=after.assume_dropped,
                nu=after.narrowing_uncertified,
            )
        )

    _restore(monkey)

    hdr = f"MUTANT={args.mutant}" if args.mutant else "REAL BUILD"
    print(f"=== ledger: {hdr} ({args.samples} oracle samples/row) ===")
    print(f"{'row':34s} {'label':10s} {'orcl':5s} {'cert':5s} "
          f"{'before -> after (per obligation)'}")
    for r in rows:
        moves = ", ".join(f"#{i}:{b}->{a}" for i, b, a in r["moves"]) or "-"
        print(f"{r['name']:34s} {r['label']:10s} "
              f"{'ne' if r['oracle_nonempty'] else 'EMP':5s} "
              f"{'Y' if r['inhabited_flag'] else '.':5s} {moves}"
              f"   [{r['verdict'][0]} -> {r['verdict'][1]}]")
    print()
    from collections import Counter

    vb = Counter(r["verdict"][0] for r in rows)
    va = Counter(r["verdict"][1] for r in rows)
    print("  verdict layer (per QUERY, reported beside the per-obligation "
          "ledger and never instead of it):")
    for s in ("VERIFIED", "REFUTED", "UNKNOWN"):
        print(f"    {s:9s} {vb[s]:3d} -> {va[s]:3d}")
    moved = [
        (r["name"], r["verdict"])
        for r in rows
        if r["verdict"][0] != r["verdict"][1]
    ]
    print(f"    queries moved: {len(moved)}; toward VERIFIED: "
          f"{sum(1 for _, v in moved if v[1] == 'VERIFIED')}")
    print()
    for k, v in tot.items():
        print(f"  {k:22s} {v}")
    return tot


if __name__ == "__main__":
    main()
