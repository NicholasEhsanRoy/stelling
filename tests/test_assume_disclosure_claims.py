# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What the assume-forwarding batch SAYS, checked against what it DOES.

Four claims, each one a sentence that was wrong or over-broad, each one
pinned here so that restoring the sentence goes red. None of them changes a
verdict; all of them change what a reader is told, and a verdict is only as
good as the sentence beside it.

1. **The bar's fallback note.** ``verdict._bar_scope`` re-slices without the
   propagation, so a script carrying a forwarded relational axiom cannot be
   reproduced by its re-derivation. The note said *"the escalation is not
   evidence about this query"* — but the escalation IS about this query; the
   re-derivation simply was not given the axioms.
2. **"Every skipped assume is disclosed in the verdict notes".** True of the
   dispatch path and false before it: escalation refused for a CONSTRAINING
   assume, for ieee semantics, or with no backend never reaches the
   disclosure loop, and a slice that declines skips it too.
3. **Two guards credited with a repair they did not make** (audit M6): the
   pairing call in ``_carry_assumes`` and the term-count raise in
   ``smt.emit``. Both are mutation-dead; what closed M6 is the scope-correct
   identity.
4. **Dead code described as live**: ``_render_scope``'s ``cond`` arm, the
   scope-not-inlined skip, and the two shape screens in ``_carry_assumes``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

jax = pytest.importorskip("jax")

from stelling import obligation as O  # noqa: E402
from stelling import propagate as P  # noqa: E402
from stelling import smt  # noqa: E402
from stelling import verdict as V  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.harness import any_array, assert_, assume  # noqa: E402
from stelling.preconditions import check  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src" / "stelling"


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """This module declares float64 inputs, so it must ask for x64 ITSELF.

    It previously inherited x64 from a bare module-scope
    `jax.config.update` in a *different* test module. pytest imports every
    test module during COLLECTION, before any test runs, so that one call
    set x64 for the whole session and every module silently rode on it.
    That call had no restore and leaked into
    `test_transcribe.py::test_content_hash_stable_across_processes`, which
    compares an in-process hash against a clean subprocess — parent f64,
    child f32, hashes differ. Removing it left this module declaring
    float64 into a float32 session, where the declarations truncate, every
    obligation declines, and the assertions fail on a
    `DeclinedObligation`. A module-scoped fixture that saves and restores
    is the house pattern and does not depend on collection order.
    """
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ---------------------------------------------------------------------------
# 1 — the bar's fallback sentence
# ---------------------------------------------------------------------------

def _assume_query():
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    assume(x < y)
    return (assert_(x - y <= 0.0),)


def test_a_slice_with_assumes_and_one_without_share_a_slice_fingerprint():
    """The measured fact the corrected sentence rests on, so the sentence is
    not the only place it is written down.

    `smt.slice_fingerprint` walks `sl.eqns` and never `sl.assumes`, so the
    slice the escalation ran on and the slice `_bar_scope` re-derives have
    the SAME `slice_sha256` and differ only in `smt2_sha256` — by the
    `(assert ...)` axiom lines. The re-derivation therefore fails to
    reproduce the SCRIPT, not the slice, and it fails for a reason that says
    nothing about whether the escalation was about this query.
    """
    closed = transcribe(jax.make_jaxpr(_assume_query)())
    p = P.propagate(closed)
    env = P.interval_env(closed)
    assert len(p.relational_assumes) == 1
    with_axioms = O.slice_obligation(
        closed, 0, env, relational_assumes=p.relational_assumes
    )
    without = O.slice_obligation(closed, 0, env)
    assert len(with_axioms.assumes) == 1 and without.assumes == ()
    a, b = smt.emit(with_axioms, "z3", 5000), smt.emit(without, "z3", 5000)
    assert a.slice_sha256 == b.slice_sha256, (
        "the fingerprint started seeing the assumes; the note's reasoning "
        "has to be re-derived if so"
    )
    assert a.sha256 != b.sha256


def test_the_bar_fallback_does_not_deny_the_escalation_is_about_the_query():
    """`_bar_scope` may say what IT could not reproduce. It may not say the
    escalation is about something else — it has not measured that, and on
    the assume-carrying shape above the claim is false."""
    src = (_SRC / "verdict.py").read_text(encoding="utf-8")
    assert "the escalation is not evidence about this query" not in src, (
        "the fallback note is asserting something _bar_scope did not measure"
    )
    assert "could not identify any recorded" in src


def _barred_assume_query():
    """The shape the corrected sentence is about: an assume-carrying query
    that also contains a barred primitive, so the bar has something to fall
    back FROM."""
    import jax.numpy as jnp
    x = any_array((3,), "float64", (-10.0, 10.0))
    y = any_array((3,), "float64", (-10.0, 10.0))
    assume(x[0] < y[0])
    s = x.at[0].set(0.5)
    return (assert_(jnp.sum(s) - jnp.sum(y) <= 100.0),)


def test_the_bar_fallback_wording_reaches_a_reader():
    """The corrected clause on a real fallback, so it is a rendered sentence
    and not only a string in the file. An empty invocation tuple is the
    simplest way to reach it: no recorded invocation can reproduce anything,
    which is the same door the assume-carrying script goes through."""
    barred, why = V._bar_scope(
        transcribe(jax.make_jaxpr(_barred_assume_query)()), {0: ()}
    )
    assert barred, "this query must carry a barred primitive or the bar is silent"
    assert "could not identify any recorded" in why
    assert "not evidence about this query" not in why


# ---------------------------------------------------------------------------
# 2 — "every skipped assume is disclosed"
# ---------------------------------------------------------------------------

def _constrained_and_skipped():
    """A CONSTRAINING assume beside a relational one whose operands no
    obligation reads. Escalation is refused before dispatch, so the skip
    reason is never produced — while the propagator's own DROPPED note is."""
    x = any_array((), "float64", (-10.0, 10.0))
    z = any_array((), "float64", (0.0, 10.0))
    w = any_array((), "float64", (0.0, 10.0))
    assume(x >= 2.0)        # CONSTRAINING: refuses escalation outright
    assume(z <= w)          # relational, forwarded, and nothing reads z or w
    return (assert_(x >= 3.0),)


def test_a_pre_dispatch_refusal_produces_no_per_assume_skip_reason():
    """The claim "every skipped assume is disclosed in the verdict notes" is
    false here, and the CHANGELOG must not make it."""
    v = check(_constrained_and_skipped, vacuity_mode="inputs-only",
              solver_timeout_ms=8000)
    rendered = v.render()
    assert "not forwarded to the solver:" not in rendered, (
        "this run reached the per-assume disclosure after all — the "
        "CHANGELOG's scoped claim can be widened again"
    )
    # nothing vanished entirely: the coarse propagator-side note is there
    assert "assume constraint DROPPED" in rendered


def test_the_changelog_scopes_the_disclosure_claim_to_the_dispatch_path():
    text = (_REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Every skipped assume is disclosed** in the verdict notes" not in text
    assert "every assume the slice declines to state" in text


def test_the_verdict_guide_scopes_the_same_claim():
    """The same sentence lives in the page a user actually reads when an
    UNKNOWN surprises them, so it is scoped in both places or in neither."""
    text = (_REPO / "docs" / "reading-a-verdict.md").read_text(encoding="utf-8")
    assert "verdict says which assume was not\n   forwarded and why" not in text
    assert "and the escalation reached the solver\n   at all**" in text
    assert "refused *before* dispatch carries no such note" in text


# ---------------------------------------------------------------------------
# 3 — two guards, credited for what they are
# ---------------------------------------------------------------------------

def test_the_pairing_call_no_longer_claims_the_M6_repair():
    """`docs/norms.md` § "Guard coverage is proven by mutation, not by
    construction". Substituting `renamed` for `probe` reddens 0 of the
    suite's collected tests, so the comment may not present it as the
    repair; the scope-correct identity is."""
    src = (_SRC / "obligation.py").read_text(encoding="utf-8")
    assert "ARITY IS CHECKED, AND `_pair_elementwise` ABOVE IS THE CHECK." in src
    assert "ABOVE IS THE CHECK\n            # (audit M6)" not in src
    assert "BELT AND BRACES BEHIND THE IDENTITY REPAIR" in src
    assert "reddens 0 of the suite's" in src


def test_the_emission_raise_no_longer_claims_the_M6_crash():
    src = (_SRC / "smt.py").read_text(encoding="utf-8")
    assert "M6'S CRASH IS NOT WHAT THESE CLOSE" in src
    assert "reddens 0 of the suite's" in src
    assert "Kept, not credited." in src


# ---------------------------------------------------------------------------
# 4 — dead code described as live
# ---------------------------------------------------------------------------

def test_no_forwarded_relational_assume_can_carry_a_cond_scope_step():
    """The fact the three "unreachable" notes rest on. `branch_depth` is
    incremented BEFORE the `("cond", pos, i)` step is pushed, and the
    forwarding guard refuses while it is nonzero."""
    import jax.numpy as jnp

    def h():
        x = any_array((), "float64", (0.0, 10.0))
        y = any_array((), "float64", (0.0, 10.0))
        s = any_array((), "float64", (0.0, 10.0))

        def yes(v):
            assume(v[0] >= v[1])
            return v[0] * 0.0

        def no(v):
            return v[1] * 0.0

        jax.lax.cond(s > 0.0, yes, no, jnp.stack([x, y]))
        return (assert_(x <= 10.0),)

    p = P.propagate(transcribe(jax.make_jaxpr(h)()))
    assert p.relational_assumes == ()
    assert all(
        step[0] != "cond"
        for ra in p.relational_assumes for step in ra.scope
    )


def test_the_unreachable_arms_say_that_they_are_unreachable():
    """A comment that presents a dead arm as live sends the next reader
    looking for a cause that does not exist."""
    src = (_SRC / "obligation.py").read_text(encoding="utf-8")
    assert "THE ``cond`` ARM IS UNREACHABLE FROM THE ONLY CALLER" in src
    assert "THE NEXT THREE SKIPS ARE SCREENED BY THE PROPAGATOR FIRST" in src
    prop = (_SRC / "propagate.py").read_text(encoding="utf-8")
    assert "NO FORWARDED ASSUME EVER CARRIES ONE OF THESE STEPS" in prop


def test_render_scope_still_renders_a_cond_step_it_is_handed():
    """Unreachable from the caller is not the same as deleted: the renderer
    is kept, and a path with a cond step still becomes prose rather than a
    `repr`."""
    assert O._render_scope((("cond", 4, 1),)) == (
        "branch 1 of the cond at equation 4"
    )
    assert O._render_scope((("call", 4),)) == "the body of equation 4"
    assert O._render_scope(()) == "the top level"
