# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Every route into a declaration must reach the SAME decision, because the
storability guard keys on the operand's own type and a route that pre-converts
its bounds hands the guard a value already rounded.

Four routes have now been found, each after the previous was closed:
`np.integer` into `any_array`, `np.longdouble` into `any_array`, `float()`
through `any_pytree`'s `canon()`, and `float()` through
`contracts._closed_range`. The first two were fixable inside the guard. **The
last two were not, because the loss happened upstream of it** — which is why
this file tests ROUTE AGREEMENT rather than any single route's behaviour.

The measured defect: the hand form refused `int64 (0, 2**53 + 1)` while the
sugar admitted it with `hi` stored as `9007199254740992.0`, so the declared
integer was not in the box the tool then reasoned about.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
import numpy as np

from stelling.harness import any_array, any_pytree

# Bound pairs spanning the decision: exact, widening (admitted), narrowing
# (refused), and ordinary. `dtype` matters — the guard is about integers that
# binary64 cannot hold.
PAIRS = [
    ("int64", 0, 2**53),                 # exact
    ("int64", 0, 2**53 + 1),             # hi rounds DOWN -> widens -> admit
    ("int64", 2**53 + 1, 2**60),         # lo rounds UP  -> narrows -> refuse
    ("int64", -(2**53) - 1, 0),          # lo rounds DOWN -> widens -> admit
    ("int64", -(2**63), 2**63 - 1),      # the natural "any int64"
    ("uint64", 0, 2**64 - 1),
    ("uint64", 2**64 - 1, 2**64 - 1),
    ("int64", 0, 10),
    ("float64", 0.0, 100.0),
    ("float32", 0.0, 1.0),
]


def _hand(dtype, lo, hi):
    try:
        jax.make_jaxpr(lambda: (any_array((1,), dtype, (lo, hi)),))()
        return "admit"
    except ValueError as e:
        return f"refuse:{str(e)[:40]}"


def _sugar(dtype, lo, hi):
    proto = np.zeros((1,), np.dtype(dtype))
    try:
        jax.make_jaxpr(lambda: any_pytree(proto, (lo, hi)))()
        return "admit"
    except ValueError as e:
        return f"refuse:{str(e)[:40]}"


@pytest.mark.parametrize("dtype,lo,hi", PAIRS)
def test_the_hand_and_sugar_routes_reach_the_same_decision(dtype, lo, hi):
    """The property, stated as agreement rather than as either route's answer,
    so it holds whichever way the guard's decision goes and fails the moment a
    route starts deciding for itself."""
    h, s = _hand(dtype, lo, hi), _sugar(dtype, lo, hi)
    assert h == s, (
        f"{dtype} ({lo!r}, {hi!r}): the hand form says {h!r} and the sugar "
        f"says {s!r}. A route that pre-processes its bounds hands the "
        f"storability guard a value already rounded, and the guard keys on the "
        f"operand's own type — so it never sees what it exists to judge."
    )


@pytest.mark.parametrize("dtype,lo,hi", PAIRS)
def test_an_admitted_declaration_stores_a_box_containing_what_was_declared(dtype, lo, hi):
        # ANTI-VACUITY on the thing that actually went wrong: not "does it
        # refuse" but "if it admits, is the declared value IN the stored box".
        # The defect admitted `hi = 2**53 + 1` and stored `2**53`.
    proto = np.zeros((1,), np.dtype(dtype))
    try:
        cj = jax.make_jaxpr(lambda: any_pytree(proto, (lo, hi)))()
    except ValueError:
        return  # refused: nothing stored, nothing to check
    params = dict(cj.eqns[0].params)
    assert params["lo"] <= lo, (
        f"stored lo={params['lo']!r} is ABOVE the declared {lo!r}: the box "
        f"excludes values that were declared"
    )
    assert params["hi"] >= hi, (
        f"stored hi={params['hi']!r} is BELOW the declared {hi!r}: the box "
        f"excludes values that were declared"
    )


def test_the_contract_template_route_does_not_pre_convert_either():
    """`contracts._closed_range` returned `float(rng[0]), float(rng[1])`, which
    both mis-declared the envelope and printed the rounded number into the
    contract's own requires-description."""
    from stelling.contracts import _closed_range
    v = 2**53 + 1
    lo, hi = _closed_range("t", "n", (0, v))
    assert hi == v and type(hi) is int, (
        f"_closed_range must return the caller's own value, got {hi!r}"
    )
    # and it still validates
    with pytest.raises(ValueError, match="empty envelope"):
        _closed_range("t", "n", (1, 0))
    with pytest.raises(ValueError, match="non-finite endpoint"):
        _closed_range("t", "n", (0, float("inf")))


def test_the_alias_comparison_still_ignores_int_versus_float_spelling():
    """`canon` no longer normalises, so this is worth pinning: python compares
    `0 == 0.0` as equal, so the same object given `(0, 1)` at one tree position
    and `(0.0, 1.0)` at another is still one declaration."""
    a = np.zeros((1,), np.float64)
    assert jax.make_jaxpr(
        lambda: any_pytree({"x": a, "y": a}, {"x": (0, 1), "y": (0.0, 1.0)})
    )() is not None
    # and genuinely different bounds on one object still refuse
    with pytest.raises(ValueError, match="aliases an earlier leaf"):
        jax.make_jaxpr(
            lambda: any_pytree({"x": a, "y": a}, {"x": (0, 1), "y": (0, 2)})
        )()


def test_no_module_pre_converts_a_bound_before_any_array_sees_it():
    """AN INVARIANT WITH A TEST IT CANNOT DRIFT PAST, rather than a grep.

    Four routes have been found, each after the previous was closed, so the
    next one is the expected outcome rather than a surprise. This makes the
    property structural: no function in `src/stelling` may hand `any_array` a
    bound it has already pushed through `float()`, and no helper whose result
    feeds `any_array` may return one.

    It is deliberately coarse — it flags the SHAPE (a `float()` on a bound
    expression) rather than proving reachability — because a false alarm here
    costs a comment and the failure it prevents cost a soundness defect three
    times.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "stelling"
    offenders = []
    for f in sorted(src.glob("*.py")):
        text = f.read_text()
        # DERIVED, not hand-listed: only a module that mentions `any_array` can
        # be on a declaration route. Without this the check flags
        # `interval.py`'s truncating-division corners, which return
        # `float(min(qs)), float(max(qs))` as interval arithmetic and have
        # nothing to do with a declared bound — measured, and it is the reason
        # this filter exists rather than a module allowlist.
        #
        # THE LIMITATION IS REAL AND STATED: a NEW declaration route in a module
        # that reaches `any_array` indirectly would escape this. Four routes have
        # been found, each after the previous was closed, so a fifth is the
        # expected outcome — and this check narrows where to look rather than
        # proving there is nowhere left.
        if "any_array" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            # (a) a bounds argument that is itself a float() call or a tuple of them
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                if name == "any_array" and len(node.args) >= 3:
                    b = node.args[2]
                    elts = b.elts if isinstance(b, (ast.Tuple, ast.List)) else [b]
                    for e in elts:
                        if (isinstance(e, ast.Call)
                                and isinstance(e.func, ast.Name)
                                and e.func.id == "float"):
                            offenders.append(f"{f.name}:{node.lineno} any_array bound is float(...)")
            # (b) `return float(a), float(b)` — the canon()/_closed_range shape
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                floats = [e for e in node.value.elts
                          if isinstance(e, ast.Call)
                          and isinstance(e.func, ast.Name) and e.func.id == "float"]
                if len(floats) == len(node.value.elts) == 2:
                    offenders.append(
                        f"{f.name}:{node.lineno} returns (float(...), float(...)) — "
                        f"the canon()/_closed_range shape"
                    )
    assert not offenders, (
        "a declaration route pre-converts its bounds:\n  "
        + "\n  ".join(offenders)
        + "\n`any_array`'s storability guard keys on the operand's own type, so "
          "a pre-converted bound arrives already rounded and the guard never "
          "sees the value it exists to judge. Pass the caller's own values "
          "through and let the guard decide — it already made this decision, "
          "and re-deciding it is how the layers came to disagree."
    )
