# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE GENERATOR FLOOR: the strategies produce the shapes they claim to.

Every property in this suite already asserts a census floor on what IT
examined. This file asserts the floor one level lower: that the *strategies*
are still drawing the boundary classes they were written to draw. The two catch
different things. A property's census can be healthy while the strategy has
quietly stopped producing, say, out-of-dtype-range constants — the property
still runs hundreds of harnesses, they are simply all boring, and it goes green
for a reason nobody can see from the outside.

That is not hypothetical for this project. The class that matters most here —
**an integer bound or constant OUTSIDE its own dtype's range** — is a class
``hypothesis.extra.numpy.from_dtype`` is structurally incapable of producing:
measured, **0 of 3000** draws, because ``from_dtype`` is dtype-faithful by
construction. It yields values *of* the dtype. The three classes it misses
entirely are the three where a value must be outside its dtype, and those are
exactly where this project's open wrap defect lives. So the hand-written pools
in ``_grammar`` are not a stylistic preference over ``extra.numpy``; they are
the only source of the input class this suite most needs, and if they degrade
there is nothing behind them.

**These tests do not need jax and do not call ``check``.** They read the
generated harness, not the tool's answer, so they are cheap enough to run at
every budget and they still fail loudly if a strategy collapses.

The floors are set well below the measured rates — they are tripwires for
collapse, not targets. The measured rates at the ``ci`` profile are recorded
beside each one so that a future drift is visible as a drift and not just as a
number that still clears a bar.
"""

from __future__ import annotations

import collections

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")

from hypothesis import given  # noqa: E402

import _grammar  # noqa: E402
import _profiles  # noqa: E402


def _classify_integer(spec, seen):
    d = spec.decls[0]
    seen[f"dtype:{d.dtype}"] += 1
    seen[f"shape:{d.shape}"] += 1
    if d.lo == d.hi:
        seen["point box"] += 1
    if d.hi - d.lo >= 16:
        seen["wide box"] += 1
    consts = _grammar.folded_constants(spec)
    if consts:
        seen["carries a constant"] += 1
    if _grammar.wrappable_constants(spec):
        seen["constant outside the dtype range"] += 1
    if any(isinstance(c, int) and abs(c) > 2**33 for c in consts):
        seen["constant beyond 2**33"] += 1
    if any(s.kind == "assume" for s in spec.stmts):
        seen["has an assume"] += 1
    else:
        seen["has no assume"] += 1
    if _grammar.exact_supported(spec):
        seen["exactly decidable"] += 1
    if _grammar.declared_points(spec) is not None:
        seen["box enumerable"] += 1
    _ops(spec, seen)


def _ops(spec, seen):
    def walk(e):
        if not isinstance(e, tuple):
            return
        if e[0] == "bin":
            seen[f"op:{e[1]}"] += 1
        elif e[0] == "un":
            seen[f"op:{e[1]}"] += 1
        elif e[0] in ("pow", "cast", "sum", "cancel", "at_add"):
            seen[f"op:{e[0]}"] += 1
        for c in e[1:]:
            walk(c)

    def walk_pred(p):
        seen[f"connective:{p[0]}"] += 1
        if p[0] == "cmp":
            seen[f"cmp:{p[1]}"] += 1
            walk(p[2])
            walk(p[3])
        elif p[0] in ("and", "or"):
            walk_pred(p[1])
            walk_pred(p[2])
        elif p[0] in ("not", "all"):
            walk_pred(p[1])

    for s in spec.stmts:
        walk_pred(s.pred)


def _require(seen, floors, label):
    short = [f"{k}={seen[k]} < {v}" for k, v in sorted(floors.items()) if seen[k] < v]
    assert not short, (
        f"GENERATOR FLOOR NOT MET for {label} — the strategy has stopped "
        f"producing shapes this suite depends on, which would make every "
        f"property over it pass for a reason nobody can see:\n  "
        + "; ".join(short)
        + "\n  observed: "
        + repr(dict(sorted(seen.items())))
    )


def test_the_integer_grammar_still_draws_its_boundary_classes():
    """Measured at ``ci`` (500 draws): every integer dtype 30-90 times; boxes
    enumerable 500/500; a constant outside the dtype range ~250; point boxes
    ~90; an assume on ~half."""
    seen = collections.Counter()

    @_profiles.current().settings(500)
    @given(_grammar.integer_specs())
    def draw(spec):
        _classify_integer(spec, seen)

    draw()
    floors = {f"dtype:{d}": 5 for d in _grammar.INT_DTYPES}
    floors.update(
        {
            "shape:()": 40,
            "shape:(2,)": 40,
            "point box": 10,
            "wide box": 10,
            "carries a constant": 100,
            # THE ONE THAT MATTERS: `extra.numpy.from_dtype` produces 0 of
            # these, and they are the whole of the open wrap defect.
            "constant outside the dtype range": 40,
            "constant beyond 2**33": 5,
            "has an assume": 40,
            "has no assume": 40,
            "exactly decidable": 200,
            "box enumerable": 200,
            "op:add": 20,
            "op:sub": 20,
            "op:mul": 20,
            "op:max": 5,
            "op:min": 5,
            "op:neg": 5,
            "op:abs": 5,
            "op:square": 5,
            "op:pow": 5,
        }
    )
    floors.update({f"cmp:{c}": 10 for c in _grammar.CMP_OPS})
    _require(seen, floors, "integer_specs")


def test_the_wrap_biased_grammar_only_draws_wrappable_harnesses():
    """The biased generator's contract, asserted rather than assumed.

    If the filter ever stopped biting, the xfail(strict) leg would be searching
    the unbiased grammar at a small budget and would XPASS — reporting the open
    wrap defect as FIXED. That is a silent-success shape with a very expensive
    consequence, so the contract is checked directly.
    """
    seen = collections.Counter()

    @_profiles.current().settings(200)
    @given(_grammar.wrap_biased_integer_specs(shapes=((),), max_leaves=4))
    def draw(spec):
        seen["drawn"] += 1
        if _grammar.wrappable_constants(spec):
            seen["wrappable"] += 1
        if spec.decls[0].shape == ():
            seen["scalar"] += 1

    draw()
    assert seen["drawn"] >= 100, f"only {seen['drawn']} draws"
    assert seen["wrappable"] == seen["drawn"], (
        f"{seen['drawn'] - seen['wrappable']} of {seen['drawn']} draws from "
        "wrap_biased_integer_specs carried NO out-of-dtype-range constant; the "
        "filter is not biting and the xfail(strict) oracle leg would be "
        "searching the wrong space"
    )
    assert seen["scalar"] == seen["drawn"], "shapes= was not honoured"


def test_the_general_grammar_still_draws_its_boundary_classes():
    """Measured at ``ci`` (400 draws): size-0 shapes ~180, infinite endpoints
    ~55, subnormal endpoints ~45, signed zero ~40, non-integral integer bounds
    ~35, float bounds outside the declared float dtype ~30, mixed rank ~90."""
    seen = collections.Counter()

    @_profiles.current().settings(400)
    @given(_grammar.general_specs())
    def draw(spec):
        shapes = {d.shape for d in spec.decls}
        if len(shapes) > 1:
            seen["mixed rank"] += 1
        if len(spec.decls) > 1:
            seen["several declarations"] += 1
        for d in spec.decls:
            seen[f"kind:{'float' if d.dtype in _grammar.FLOAT_DTYPES else ('bool' if d.dtype == 'bool' else 'int')}"] += 1
            if _grammar.n_elements(d.shape) == 0:
                seen["size-0 shape"] += 1
            if len(d.shape) > 1:
                seen["rank > 1"] += 1
            for v in (d.lo, d.hi):
                if isinstance(v, float):
                    if v in (float("inf"), float("-inf")):
                        seen["infinite endpoint"] += 1
                    elif v != 0.0 and abs(v) < 2.2250738585072014e-308:
                        seen["subnormal endpoint"] += 1
                    elif v == 0.0 and str(v).startswith("-"):
                        seen["signed zero"] += 1
                    if d.dtype in _grammar.INT_DTYPES:
                        seen["non-integral integer bound"] += 1
                if isinstance(v, int) and not _grammar.in_dtype_range(d.dtype, v):
                    seen["integer bound outside the dtype range"] += 1
        if _grammar.n_elements(spec.decls[0].shape) > 0 and any(
            _grammar.n_elements(d.shape) == 0 for d in spec.decls
        ):
            seen["size-0 beside non-empty"] += 1
        if len(spec.stmts) > 1:
            seen["several statements"] += 1
        if any(s.kind == "assume" for s in spec.stmts):
            seen["has an assume"] += 1
        _ops(spec, seen)

    draw()
    _require(
        seen,
        {
            "kind:float": 100,
            "kind:int": 100,
            "kind:bool": 5,
            "size-0 shape": 30,
            "rank > 1": 10,
            "mixed rank": 20,
            "several declarations": 100,
            # The shape the size-0 conjunct defect needs.
            "size-0 beside non-empty": 10,
            "infinite endpoint": 5,
            "subnormal endpoint": 5,
            "integer bound outside the dtype range": 20,
            "non-integral integer bound": 3,
            "several statements": 100,
            "has an assume": 50,
            "op:add": 20,
            "op:mul": 20,
            "op:cancel": 5,
            "op:sum": 5,
            "op:cast": 5,
            "op:sign": 3,
            "connective:and": 10,
            "connective:or": 10,
            "connective:not": 10,
        },
        "general_specs",
    )


def test_the_cross_series_corpus_still_contains_what_its_control_needs():
    """The deterministic corpus, checked structurally rather than by running it.

    ``_corpus`` is not a Hypothesis strategy — it is a fixed pseudo-random list,
    so it can be reproduced in two interpreters at once. That also means a
    change to the pools silently reshapes it, and the one entry its positive
    control depends on is hand-written rather than drawn. Both are asserted.
    """
    import _corpus

    specs = _corpus.corpus()
    assert len(specs) >= 200, f"corpus collapsed to {len(specs)} harnesses"
    rendered = "\n".join(s.render() for s in specs)
    assert ".at[" in rendered, (
        "the corpus no longer contains a scatter-add harness; the cross-series "
        "positive control at 8ef8f75 keys on exactly that primitive and would "
        "stop firing"
    )
    seen = collections.Counter()
    for s in specs:
        for d in s.decls:
            seen[d.dtype] += 1
        _ops(s, seen)
    assert len({k for k in seen if k in _grammar.ALL_DTYPES}) >= 8, (
        f"the corpus covers only {sorted(k for k in seen if k in _grammar.ALL_DTYPES)}"
    )
    # Reproducibility, which is the whole basis of the two-interpreter diff.
    assert [s.render() for s in _corpus.corpus()] == [s.render() for s in specs], (
        "_corpus.corpus() is not deterministic within one interpreter, so it "
        "cannot be compared across two"
    )
