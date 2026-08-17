# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""AUDIT 0.2.0 S12, end to end through the documented deserialization door.

`stelling.harness` cannot build this query — jax refuses to trace an
ill-typed `dot_general` — but `ir.ClosedJaxpr.from_dict` can load one, and
`ir.py`'s own module docstring puts full per-primitive shape inference
EXPLICITLY out of that door's scope, naming "the in-pipeline read gate and
shape/decode predicates" as the defence past its bounded set. So this is
the route the defect was reachable by, and the in-pipeline predicate is
where the repair had to go.

The query: four declared elements in `[1, 2]`, contracted against the
constant `[1, 1, 1, 1]`, asserting the sum is `<= 4.5`. The DECLARATION is
then edited to shape `(2,)` while the constant operand stays `(4,)`, and
the document is reloaded.

Measured on `main` (`dee8bc2`), jax 0.11.0, `JAX_ENABLE_X64=1`, z3 + cvc5
wheels:

    from_dict ACCEPTED the mismatched query
    propagation obligations: [(0, 'unknown', '... the operand spans
                              [-inf, inf] ...')]
    ESCALATION assert #0 -> discharged
       detail: discharged by solver escalation (QF_LRA): the box with the
               negated predicate is unsat per z3 (wheel) and cvc5 (wheel)

A VERIFIED, over a truncated two-term reading of a four-term contraction.
The truth, in exact rationals and computed here rather than asked of the
tool: over the four-term reading the sum lies in `[4, 8]`, so `<= 4.5` is
NOT valid; over the truncated two-term reading it lies in `[2, 4]` and the
threshold holds. That gap is the false verdict.
"""
from __future__ import annotations

import copy
from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling import _optional  # noqa: E402
from stelling import ir  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.solvers import OB_DISCHARGED, SolverConfig, escalate  # noqa: E402

HAVE_Z3 = _optional.available("z3")
HAVE_CVC5 = _optional.available("cvc5") or _optional.cvc5_binary() is not None


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


LO, HI = Fraction(1), Fraction(2)
THRESHOLD = Fraction(9, 2)


def _harness():
    a = any_array((4,), jnp.float64, (float(LO), float(HI)))
    c = jnp.array([1.0, 1.0, 1.0, 1.0])
    return assert_(jnp.dot(a, c) <= float(THRESHOLD))


def _shrink_the_declaration_only(d):
    """Edit the DECLARATION (and the dot_general's lhs invar aval) to shape
    `(2,)`, leaving the constant operand at its four elements. Returns the
    number of edits made, so a serialization change that silently stops
    matching is a failure rather than a vacuous pass."""
    edits = 0

    def walk(o):
        nonlocal edits
        if isinstance(o, dict):
            if o.get("primitive") == "stelling_any":
                o["params"] = [
                    ["shape", {"k": "tuple", "items": [2]}]
                    if p[0] == "shape"
                    else p
                    for p in o["params"]
                ]
                for ov in o["outvars"]:
                    ov["aval"]["shape"] = [2]
                edits += 1
            if o.get("primitive") == "dot_general":
                o["invars"][0]["aval"]["shape"] = [2]
                edits += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(d)
    return edits


def _mismatched_query():
    d = copy.deepcopy(trace(_harness).to_dict())
    assert _shrink_the_declaration_only(d) == 2, (
        "the serialization no longer has the two nodes this edit targets; "
        "the test would measure nothing"
    )
    return ir.ClosedJaxpr.from_dict(d)


def test_the_exact_truth_about_this_query_needs_no_verifier():
    """The oracle, in exact rationals, stated before the tool is asked."""
    four_term_max = 4 * HI
    two_term_max = 2 * HI
    assert four_term_max == Fraction(8) and two_term_max == Fraction(4)
    assert not (four_term_max <= THRESHOLD), (
        "the four-term claim is NOT valid, so no VERIFIED about it is sound"
    )
    assert two_term_max <= THRESHOLD, (
        "the truncated two-term claim IS valid — which is why dropping two "
        "addends produced a VERIFIED"
    )


def test_the_door_still_accepts_the_mismatched_document():
    """UNCHANGED, and deliberately so.

    The repair is not a door check. `ir.py` scopes per-primitive shape
    inference out of `_validate_loaded` in writing, and a rule added there
    for this one primitive would leave the two faces still able to disagree
    on any query built another way — `ClosedJaxpr` is a public dataclass and
    `smt.emit` takes hand-built IR. Pinning that the door is unchanged is
    what stops a later reader "fixing" this at the door and concluding the
    in-pipeline oracle is redundant.
    """
    q = _mismatched_query()
    eqn = next(e for e in q.jaxpr.eqns if e.primitive == "dot_general")
    assert tuple(eqn.invars[0].aval.shape) == (2,)
    assert tuple(eqn.invars[1].aval.shape) == (4,)


@pytest.mark.skipif(
    not (HAVE_Z3 and HAVE_CVC5), reason="needs both z3 and cvc5"
)
def test_escalation_no_longer_discharges_the_truncated_contraction():
    """THE FALSE VERIFIED, gone — and gone by DECLINING, not by answering.

    The propagation leg still lands the obligation at `unknown` (its
    transfer refuses the equation, so the value is ⊤), which is what routes
    it to escalation. What changed is that escalation now finds nothing it
    can plan: the emission asks the same shape oracle the transfer asked,
    gets the same refusal, and quotes it.
    """
    q = _mismatched_query()
    prop = propagate(q)
    assert [o.status for o in prop.obligations] == ["unknown"]

    esc = escalate(q, prop, SolverConfig(timeout_ms=10_000))
    (record,) = esc.records
    assert record.outcome != OB_DISCHARGED, (
        "the truncated contraction was discharged again: " f"{record.detail}"
    )
    assert "contracted dims disagree" in record.detail, record.detail
    assert "lhs[0]=2 vs rhs[0]=4" in record.detail, record.detail


def test_the_other_direction_declines_instead_of_crashing():
    """S12's second half: the SHORTER CONSTANT operand.

    With the constant operand shorter than the symbolic one the plan
    indexed off its end and raised a raw `IndexError` — out of
    `slice_obligation`, which catches only `_Decline`, and out of
    `escalate`, whose own `except Exception` net sits INSIDE the loop body
    and not around `slice_unknown_obligations` in the `for` header. So one
    ill-typed equation took every other obligation's verdict with it.

    Here the constant is `(4,)` and the declaration is `(2,)`, which makes
    the SYMBOLIC operand the short one — the truncating direction; the
    crashing direction is covered without jax in
    `test_dot_general_both_faces.py`, where the plan is driven directly.
    What this pins is the property that matters at this level: escalation
    RETURNS, with a record, for every obligation.
    """
    q = _mismatched_query()
    prop = propagate(q)
    esc = escalate(q, prop, SolverConfig(timeout_ms=10_000))
    assert len(esc.records) == len(prop.obligations)
