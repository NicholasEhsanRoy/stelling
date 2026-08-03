# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The element-budget decline gates the ATTEMPT and says so.

docs/proposed-decline-messages.md #4: the old sentence was accurate,
quantitative — and read as a diagnosis, sending readers to tune
ELEMENT_BUDGET. The decline now says "not attempted", names WHICH
quantity exceeded the budget (bound to the gate's own comparison), states
that the budget bounds only what escalation will attempt (the interval
result is budget-independent by construction — measured here), and names
the lever that actually lowers the counts (a smaller obligation). The
proposal's "obligations past the budget declined again on a different
cause once it was raised" is a claim about unrecorded measurements and is
deliberately NOT in the message; nothing here states it.

Message content only: the gate's decision is untouched (same trigger,
same DeclinedObligation), and the pre-existing quantity/budget pins in
test_array_emission/test_contracts still hold verbatim.
"""

from __future__ import annotations

import stelling.obligation as ob
from stelling.obligation import (
    ELEMENT_BUDGET,
    DeclinedObligation,
    ObligationSlice,
    slice_obligation,
)
from stelling.propagate import interval_env, propagate
from test_array_emission import aval
from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, var


def _wide_query(n):
    """x in [-1, 1]^n, assert x > 0: n input terms + n comparison terms =
    2n element terms, n root conjuncts."""
    x = var(0, aval((n,)))
    pred = var(1, aval((n,), "bool"))
    out = var(2, aval((n,), "bool"))
    return close(
        [
            any_eqn(x, -1.0, 1.0, shape=(n,)),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _declined(q):
    item = slice_obligation(q, 0, interval_env(q))
    assert isinstance(item, DeclinedObligation)
    return item.reason


def test_terms_only_overage_names_element_terms_with_both_counts():
    n = ELEMENT_BUDGET  # terms 2n > budget; root n == budget, NOT over
    reason = _declined(_wide_query(n))
    assert "obligation not attempted:" in reason
    assert f"it needs {2 * n} element terms and {n} root conjuncts" in reason
    # the exceeding quantity is named by the same comparison the gate ran
    assert 2 * n > ELEMENT_BUDGET >= n
    assert (
        f"its element terms put it over the per-obligation emission "
        f"budget of {ELEMENT_BUDGET}" in reason
    )
    assert "root conjuncts put it over" not in reason


def test_root_only_overage_names_root_conjuncts():
    # a scalar predicate broadcast to a huge root: 2 element terms,
    # n root conjuncts — the opposite direction
    n = 4 * ELEMENT_BUDGET
    a = var(0)
    p = var(1, BOOL)
    b = var(2, aval((n,), "bool"))
    out = var(3, aval((n,), "bool"))
    q = close(
        [
            any_eqn(a, 0.0, 1.0),
            eqn("le", [a, lit(0.5)], p),
            eqn(
                "broadcast_in_dim",
                [p],
                b,
                [("shape", (n,)), ("broadcast_dimensions", ())],
            ),
            eqn("stelling_assert", [b], out),
        ],
        [out],
    )
    reason = _declined(q)
    assert f"it needs 2 element terms and {n} root conjuncts" in reason
    assert 2 <= ELEMENT_BUDGET < n
    assert (
        f"its root conjuncts put it over the per-obligation emission "
        f"budget of {ELEMENT_BUDGET}" in reason
    )
    assert "element terms put it over" not in reason
    assert "element terms and root conjuncts put it over" not in reason


def test_both_over_names_both():
    n = 2 * ELEMENT_BUDGET  # terms 4B, root 2B: both over
    reason = _declined(_wide_query(n))
    assert (
        f"its element terms and root conjuncts put it over the "
        f"per-obligation emission budget of {ELEMENT_BUDGET}" in reason
    )


def test_the_budget_gates_the_attempt_and_not_the_interval_result(
    monkeypatch,
):
    # the message's two mechanism claims, measured:
    n = ELEMENT_BUDGET
    q = _wide_query(n)
    # (1) the interval result is budget-independent: the propagation is
    # identical under a huge budget — the budget is not read there
    p_before = propagate(q)
    monkeypatch.setattr(ob, "ELEMENT_BUDGET", 10**9)
    p_after = propagate(q)
    assert p_before == p_after
    assert p_before.obligations[0].status == "unknown"
    # (2) the budget bounds what escalation will ATTEMPT: with the budget
    # raised, the same obligation is attempted (a real slice), so the
    # decline was the budget's doing and nothing else's
    item = slice_obligation(q, 0, interval_env(q))
    assert isinstance(item, ObligationSlice)
    monkeypatch.setattr(ob, "ELEMENT_BUDGET", ELEMENT_BUDGET)
    item2 = slice_obligation(q, 0, interval_env(q))
    assert isinstance(item2, DeclinedObligation)


def test_a_smaller_declared_array_lowers_the_quoted_counts(monkeypatch):
    # the named lever, measured: halving the declared array halves both
    # quoted quantities (the counts are per-element by construction)
    monkeypatch.setattr(ob, "ELEMENT_BUDGET", 4)  # keep both cases declining
    n = 16
    big = _declined(_wide_query(n))
    small = _declined(_wide_query(n // 2))
    assert f"it needs {2 * n} element terms and {n} root conjuncts" in big
    assert (
        f"it needs {n} element terms and {n // 2} root conjuncts" in small
    )
