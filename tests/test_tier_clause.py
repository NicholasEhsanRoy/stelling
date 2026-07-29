# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""`TIER_EXACT` says the returned box IS the hull of the achievable image, not
merely a sound enclosure of it — and the tier rides into the verdict stamp.

It was asserted in prose for every row and checked by nothing, and a blinded
audit found `sign` claiming it falsely by *reading* the claim rather than by
running anything. This is the clause that runs it.

IT ONLY EVER PROVES, NEVER REFUTES, and that direction is the whole design.
With `S` a set of sampled image points, `H(S)` its hull, `I` the true image
hull and `B` the returned box, soundness gives `H(S) ⊆ I ⊆ B`. So:

* **`H(S) = B` implies `H(S) = I = B`: the row is EXACT on that box, proven.**
* **`H(S) ⊊ B` implies nothing.** The samples may simply have missed an extremum.

A first prototype ran the refuting direction on a `linspace` grid and refuted
`abs([-3, 3])`, which is genuinely exact, because the grid never landed on 0.
Built on the confirming direction the clause cannot produce a false alarm;
built on the refuting one it would send a reader chasing ghosts.

SEEDING is therefore load-bearing, and it comes from each row's own BRANCH
BOUNDARIES rather than from a grid — the value where its behaviour changes,
which is where a piecewise-monotone function's extremes live. That is this
campaign's own norm ("the enumeration is built from the domain, not from the
values the author pictured") applied to the checker.

SCOPE, stated because the clause cannot cover the claim it checks:

* **Per-box proof; the universal claim stays untested.** `TIER_EXACT`
  quantifies over every box. This proves it for the boxes seeded. It is a
  battery like every other gauge here, and **a per-box proof is NOT evidence
  for the tier** — a row can be exact on one box and not another, which is
  exactly what `square` does.
* **Unary elementwise numeric rows only.** For a binary row the extremes of a
  monotone function sit at the corners, but `rem` is monotone in neither
  argument, so corner evaluation does not find them. Data-movement rows are
  exact structurally (every output element IS an input element) and a
  routing-aware extension is possible but not built here.
* **It cannot check `sound` or `sound-libm`.** Those claim enclosure, which
  sampling can only fail to contradict.
* **A missing breakpoint yields a false FAIL, never a false PASS** — so the
  failure direction is toward noise at authoring time, not toward certifying a
  wrong tier. The failure message says so, because one cause is
  soundness-adjacent and the other is a local test-authoring fix.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy as np
from jax import lax

from stelling import interval as iv
from stelling import propagate as P
from stelling._jax_compat import transcribe


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# Each row declares where its behaviour changes. These are the `elif`
# boundaries in the transfer, read from it — not guessed.
BREAKPOINTS = {
    "abs": (0.0,),                                   # the sign change
    "neg": (),                                       # monotone throughout
    "square": (0.0,),                                # the even-power minimum
    "integer_pow": (0.0,),                           # same
    "sqrt": (),                                      # monotone on its domain
    "exp": (),                                       # monotone
    "sign": (0.0, iv.MIN_NORMAL, -iv.MIN_NORMAL),    # the definite branches
}

FNS = {
    "abs": lambda x: lax.abs(x),
    "neg": lambda x: lax.neg(x),
    "square": jnp.square,
    "integer_pow": lambda x: x ** 2,
    "sqrt": jnp.sqrt,
    "exp": jnp.exp,
    "sign": lax.sign,
}

# Boxes per row: each straddles, sits above and sits below its breakpoints,
# plus one ordinary interval. `sqrt` is domain-restricted.
BOXES = {
    "abs": [(-3.0, 3.0), (1.0, 2.0), (-2.0, 0.0), (0.0, 0.0)],
    "neg": [(-3.0, 3.0), (1.0, 2.0), (0.0, 0.0)],
    "square": [(-2.0, 3.0), (1.0, 2.0), (0.1, 0.3)],
    "integer_pow": [(-2.0, 3.0), (1.0, 2.0), (0.1, 0.3)],
    "sqrt": [(1.0, 4.0), (0.0, 1.0), (2.0, 9.0)],
    "exp": [(0.0, 1.0), (-1.0, 1.0)],
    "sign": [(-3.0, 3.0), (0.5, 3.0), (0.0, 0.0),
             (5e-324, 1e-320), (iv.MIN_NORMAL, 1.0)],
}

NO_PROOF = "no-proof"
PROVEN = "proven-exact"
DECLINED = "declined"


def _eqn(prim):
    cj = transcribe(jax.make_jaxpr(FNS[prim])(jnp.zeros((1,), jnp.float64)))
    return [e for e in cj.jaxpr.eqns if str(e.primitive) == prim][0]


def _seeds(prim, lo, hi):
    """Endpoints plus every declared breakpoint inside the box. Nothing else —
    a grid is what produced the false refutation."""
    return sorted({lo, hi} | {b for b in BREAKPOINTS[prim] if lo <= b <= hi})


def clause(prim, lo, hi):
    """(verdict, transfer box, seeded image hull, seeds). PROVEN is a proof;
    NO_PROOF is not a refutation."""
    eqn = _eqn(prim)
    tf, _tier = P.TRANSFERS[prim]
    try:
        out = tf(eqn, dict(eqn.params_dict()),
                 [iv.IntervalArray(shape=(1,), los=(lo,), his=(hi,))])
    except iv.IntervalError:
        return DECLINED, None, None, ()
    if out is None:
        return DECLINED, None, None, ()
    box = (out[0].los[0], out[0].his[0])
    seeds = _seeds(prim, lo, hi)
    img = [float(np.asarray(FNS[prim](jnp.asarray(np.array([s], np.float64))))[0])
           for s in seeds]
    hull = (min(img), max(img))
    return (PROVEN if box == hull else NO_PROOF), box, hull, tuple(seeds)


def _diagnosis(prim, lo, hi, box, hull, seeds):
    """BOTH CAUSES, because they need different responses: one is a defect in
    the row, the other is a local fix to this file."""
    return (
        f"{prim!r} is registered TIER_EXACT but its box on [{lo!r}, {hi!r}] is "
        f"NOT the seeded image hull:\n"
        f"    transfer returned : [{box[0]!r}, {box[1]!r}]\n"
        f"    seeded image hull : [{hull[0]!r}, {hull[1]!r}]\n"
        f"    seeds used        : {seeds}\n"
        f"  EITHER the row is not exact on this box — a claim the verdict stamp "
        f"carries, so a soundness-adjacent defect — OR this box's seed set "
        f"misses an extremum of the image, which is a defect in "
        f"BREAKPOINTS[{prim!r}] and a local fix. A missing breakpoint can only "
        f"produce THIS failure, never a false pass, so rule that out first: "
        f"add the value where the row's behaviour changes and re-run."
    )


@pytest.mark.parametrize(
    "prim,lo,hi",
    [(p, lo, hi) for p in sorted(BOXES) for (lo, hi) in BOXES[p]
     if P.TRANSFERS[p][1] == "exact"],
)
def test_every_exact_row_is_proven_exact_on_its_seeded_boxes(prim, lo, hi):
    verdict, box, hull, seeds = clause(prim, lo, hi)
    assert verdict != DECLINED, f"{prim} declined [{lo}, {hi}]; the box is unusable"
    assert verdict == PROVEN, _diagnosis(prim, lo, hi, box, hull, seeds)


def test_the_clause_DISCRIMINATES():
    """A clause that proves everything proves nothing. At least one row
    registered `sound` must come back NO_PROOF — otherwise this file is not
    distinguishing tiers, it is just executing transfers."""
    sound_rows = [p for p in BOXES if P.TRANSFERS[p][1] in ("sound", "sound-libm")]
    assert sound_rows, "no non-exact row is covered, so nothing can discriminate"
    unproven = {
        p: [(lo, hi) for (lo, hi) in BOXES[p] if clause(p, lo, hi)[0] == NO_PROOF]
        for p in sound_rows
    }
    assert any(v for v in unproven.values()), (
        f"every covered `sound` row was proven exact on every box: {unproven}. "
        f"Either those tiers are understated or this clause cannot tell the "
        f"difference — and a clause that cannot distinguish is worse than none."
    )


def test_a_row_that_is_not_exact_must_fail_the_clause(monkeypatch):
    """ANTI-VACUITY. Doctor a row registered EXACT into one that is not, and
    the clause must catch it. Registry restored by monkeypatch."""
    orig_fn, orig_tier = P.TRANSFERS["abs"]

    def widened(eqn, params, ins):
        outs = orig_fn(eqn, params, ins)
        if outs is None:
            return None
        # one ulp too wide on the top: sound, and no longer the hull
        return [iv.IntervalArray(
            shape=outs[0].shape,
            los=outs[0].los,
            his=tuple(float(np.nextafter(h, np.inf)) for h in outs[0].his),
        )]

    monkeypatch.setitem(P.TRANSFERS, "abs", (widened, orig_tier))
    assert P.TRANSFERS["abs"][1] == "exact", "the doctored row still claims exact"
    verdicts = [clause("abs", lo, hi)[0] for (lo, hi) in BOXES["abs"]]
    assert NO_PROOF in verdicts, (
        f"a row registered EXACT whose box is one ulp too wide must fail the "
        f"clause on at least one seeded box; got {verdicts}. If it does not, "
        f"this file cannot detect an overstated tier."
    )


def test_the_seeding_is_what_makes_the_confirming_direction_reachable():
    """The prototype's failure, pinned: without the breakpoint, `abs([-3, 3])`
    cannot be proven exact, because no endpoint attains the image's minimum."""
    verdict, box, hull, seeds = clause("abs", -3.0, 3.0)
    assert verdict == PROVEN and 0.0 in seeds

    # the same box with the breakpoint withheld — a grid's worth of endpoints
    endpoints_only = sorted({-3.0, 3.0})
    img = [float(np.asarray(lax.abs(jnp.asarray(np.array([s], np.float64))))[0])
           for s in endpoints_only]
    assert (min(img), max(img)) != box, (
        "with 0 withheld the seeded hull must NOT equal the box — that gap is "
        "why a refuting clause produced a false alarm here"
    )
