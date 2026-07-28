# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""stelling.fidelity — the gauging loop's refusals, table, and render.

Deliberately jax-free with NO importorskip: the loop is pure
orchestration and must run identically in both venvs. Subjects here are
plain values; gates are predicates over them — the loop never looks
inside either.
"""

from __future__ import annotations

import pytest

from stelling.fidelity import FidelityReport
from stelling.fidelity import gauge as _gauge

# Every call below is about a refusal or a table, not about scope, so they share
# one declaration. `scope` is required and undefaulted at the real entry point --
# test_scope_is_required_and_undefaulted is what pins that.
_SCOPE = "the toy integer gates in this module; no stelling surface is driven"


def gauge(*args, scope=_SCOPE, **kwargs):
    return _gauge(*args, scope=scope, **kwargs)

# A tiny concrete stack used throughout: subjects are ints, the baseline
# is 0, gates check congruences that 0 satisfies.
GATES = {
    "even": lambda s: s % 2 == 0,
    "small": lambda s: abs(s) < 10,
}


def test_refusal_empty_mutations_is_ungauged():
    with pytest.raises(ValueError, match="no battery is ungauged"):
        gauge(0, GATES, {}, residual={})


def test_refusal_empty_gates():
    with pytest.raises(ValueError, match="empty stack gauges nothing"):
        gauge(0, {}, {"m": 1}, residual={})


def test_refusal_non_callable_gate():
    with pytest.raises(ValueError, match="not callable"):
        gauge(0, {"even": GATES["even"], "broken": "grep"}, {"m": 1},
              residual={})


def test_refusal_baseline_failing_a_gate_names_it():
    # L15 counterfactual: an implementation that skips baseline
    # validation would happily gauge a stack whose gates the TRUE subject
    # cannot pass — every mutation then looks "caught" by a gate that
    # catches everything, and the report blesses a stack that measures
    # nothing. This test fails against that implementation.
    with pytest.raises(ValueError, match=r"BASELINE fails gate\(s\) \['even'\]"):
        gauge(1, GATES, {"m": 2}, residual={})


def test_refusal_unexplained_survivor_names_it():
    # 4 passes both gates (even, small) and is not residual-listed
    with pytest.raises(ValueError, match=r"\['sneaky'\] survive every gate"):
        gauge(0, GATES, {"odd": 3, "sneaky": 4}, residual={})


def test_refusal_stale_residual_claim_names_mutation_and_gates():
    # L15 counterfactual: an implementation that treats residual as
    # ADVISORY (documentation riding along unmeasured) would return a
    # report whose residual section claims 'odd' is value-identical
    # while the table right above it shows 'odd' CAUGHT — two sections
    # of one report contradicting each other. This test fails against
    # that implementation: a caught mutation's residual entry must
    # refuse, naming both the mutation and the catching gate.
    with pytest.raises(ValueError, match=r"stale residual.*'odd' caught by \['even'\]"):
        gauge(0, GATES, {"odd": 3}, residual={"odd": "should never be caught"})


def test_refusal_residual_naming_no_known_mutation():
    with pytest.raises(ValueError, match=r"\['ghost'\].*no known mutation"):
        gauge(0, GATES, {"odd": 3}, residual={"ghost": "not in the battery"})


def test_refusal_residual_with_empty_explanation():
    with pytest.raises(ValueError, match="no explanation"):
        gauge(0, GATES, {"even_survivor": 4}, residual={"even_survivor": "  "})


def test_happy_path_table_and_residual():
    report = gauge(
        0,
        GATES,
        {"odd": 3, "big": 20, "big_odd": 21, "twin": 4},
        residual={"twin": "even and small like the baseline: value-identical"},
    )
    assert isinstance(report, FidelityReport)
    assert report.gates == ("even", "small")
    assert report.caught_by == (
        ("odd", ("even",)),
        ("big", ("small",)),
        ("big_odd", ("even", "small")),
        ("twin", ()),
    )
    assert report.residual == (
        ("twin", "even and small like the baseline: value-identical"),
    )


def test_report_is_frozen():
    report = gauge(0, GATES, {"odd": 3}, residual={})
    with pytest.raises(Exception):
        report.gates = ("tampered",)


def test_render_shows_table_and_residual_section():
    report = gauge(
        0,
        GATES,
        {"odd": 3, "twin": 4},
        residual={"twin": "value-identical to the baseline under both gates"},
    )
    text = report.render()
    assert "baseline: passes every gate (even, small)" in text
    assert "odd" in text and "CAUGHT by even" in text
    assert "survives every gate (residual, explained below)" in text
    assert "== residual class" in text
    assert "twin: value-identical to the baseline under both gates" in text


def test_render_with_empty_residual_says_so():
    report = gauge(0, GATES, {"odd": 3}, residual={})
    assert "(empty: every mutation was caught)" in report.render()


def test_gate_exceptions_propagate_not_swallowed():
    """A gate that raises is a broken gate, not a catch: converting a
    raise into 'caught' would let a crashing gate masquerade as
    discriminating power (the gate never measured anything)."""

    def exploding(subject):
        raise RuntimeError("gate blew up")

    with pytest.raises(RuntimeError, match="gate blew up"):
        gauge(0, {"even": GATES["even"], "boom": exploding}, {"m": 3},
              residual={})


def test_battery_order_is_preserved_in_table():
    report = gauge(
        0,
        GATES,
        {"z_first": 3, "a_second": 20},
        residual={},
    )
    assert [name for name, _ in report.caught_by] == ["z_first", "a_second"]


def test_scope_is_required_and_undefaulted():
    """Omitting it must be a construction error, not a silent empty claim.

    The norm this enforces was earned: a scatter gauge reported zero survivors
    while never driving the interval transfer the change under test had
    altered, and nothing in its output said so.
    """
    with pytest.raises(TypeError):
        _gauge(0, GATES, {"odd": 3}, residual={})  # no scope=


@pytest.mark.parametrize("bad", ["", "   ", None, 7])
def test_a_blank_or_non_string_scope_is_refused(bad):
    with pytest.raises(ValueError, match="no scope"):
        _gauge(0, GATES, {"odd": 3}, residual={}, scope=bad)


def test_the_scope_reaches_the_rendered_output():
    """A report that does not carry its scope gets quoted past it."""
    report = _gauge(0, GATES, {"odd": 3}, residual={},
                    scope="emission face only; the transfer is NOT driven")
    text = report.render()
    assert "emission face only; the transfer is NOT driven" in text
    # once beside the header and once beside the counts, so neither a reader
    # who skims the top nor one who skims the bottom can miss it
    assert text.count("emission face only; the transfer is NOT driven") == 2
