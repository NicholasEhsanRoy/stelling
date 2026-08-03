# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The reproducer's refusals, its schema, and the rule it exists for.

``tests/test_reproduce_acceptance.py`` measures what an emitted file does
when it runs. This file measures what the emitter refuses to do, which is
the half that has to hold when nobody is running anything: the
no-stelling-import rule checked on the artefact, the published sidecar
schema, the mandatory reachability face, and the content-hash gate that
stops a verdict being quoted over a program it is not about.
"""

from __future__ import annotations

import re

import pytest

import stelling.reproduce as R
from stelling.reproduce import (
    CONFIRMED,
    DIVERGED,
    EXECUTION_MODES,
    EXECUTION_RESULTS,
    NOT_EXECUTED_EXIT,
    RESULT_EXIT,
    SCHEMA,
    SIDECAR_KEYS,
    UNREACHABLE,
    ReproducerError,
    Subject,
)


def _fn(a):
    return a, 1.0


def _pred(a):
    return True


def _subject(**kw):
    base = dict(
        name="s",
        fn=_fn,
        relation="<=",
        declarations=(((), "float64", (0.0, 1.0)),),
        no_precondition_reason="every point of the envelope is producible",
    )
    base.update(kw)
    return Subject(**base)


# ── the tokens, and what they may never be ───────────────────────────────────


def test_the_execution_results_are_three_and_are_not_verdict_statuses():
    """A tally that meets one of these must not be able to read it as a
    verdict — the same reasoning behind the contracts layer's DECLARED."""
    assert EXECUTION_RESULTS == (CONFIRMED, DIVERGED, UNREACHABLE)
    assert len(set(EXECUTION_RESULTS)) == 3
    assert not set(EXECUTION_RESULTS) & {
        "VERIFIED", "REFUTED", "UNKNOWN", "DECLINED", "DECLARED"
    }


def test_every_result_exits_zero_and_only_the_no_result_case_does_not():
    """DIVERGED must never render as a failed check, and an exit status is
    the crudest rendering there is."""
    assert RESULT_EXIT == 0
    assert NOT_EXECUTED_EXIT != 0


# ── the mandatory reachability face ──────────────────────────────────────────


def test_a_subject_must_answer_the_reachability_question_one_way_or_other():
    with pytest.raises(ReproducerError, match="no safe default"):
        Subject(
            name="s", fn=_fn, relation="<=",
            declarations=(((), "float64", (0.0, 1.0)),),
        )
    with pytest.raises(ReproducerError, match="no safe default"):
        _subject(precondition=_pred, precondition_reason="because")


def test_a_declared_precondition_must_carry_the_argument_that_licenses_it():
    """The two episodes' lesson, structuralized: UNREACHABLE never travels
    without its reason, so the reason cannot be omitted."""
    with pytest.raises(ReproducerError, match="row7 and RigidBody"):
        Subject(
            name="s", fn=_fn, relation="<=",
            declarations=(((), "float64", (0.0, 1.0)),),
            precondition=_pred,
        )
    ok = Subject(
        name="s", fn=_fn, relation="<=",
        declarations=(((), "float64", (0.0, 1.0)),),
        precondition=_pred,
        precondition_reason="structural: the quantity is unit by construction",
    )
    assert ok.precondition is _pred


def test_a_reason_may_not_forge_a_result_line():
    """Printed at column 0 of the reproducer's output, so an embedded
    newline could manufacture an '== CONFIRMED' — the refusal EnsuresFace
    already makes, for the same reason."""
    with pytest.raises(ReproducerError, match="single physical line"):
        _subject(no_precondition_reason="fine\n== CONFIRMED")
    with pytest.raises(ReproducerError, match="single physical line"):
        Subject(
            name="s", fn=_fn, relation="<=",
            declarations=(((), "float64", (0.0, 1.0)),),
            precondition=_pred,
            precondition_reason="fine\n== UNREACHABLE",
        )


# ── the other authoring refusals ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(name=""), "name must be populated"),
        (dict(fn=3), "must be callable"),
        (dict(relation="=="), "must be one of"),
        (dict(relation="!="), "must be one of"),
        (dict(declarations=()), "no envelope here"),
        (dict(declarations=((), "float64", (0, 1))), "triple"),
        (dict(declarations=((((),), "int32", (0, 1)),)), "float32"),
        (dict(declarations=((((),), "float64", (0, 1, 2)),)), "\\(lo, hi\\) pair"),
    ],
)
def test_authoring_refusals(kw, match):
    with pytest.raises(ReproducerError, match=match):
        _subject(**kw)


def test_the_four_relations_are_exactly_the_ones_a_file_can_write_down():
    assert sorted(R._RELATIONS) == ["<", "<=", ">", ">="]


# ── what cannot be named in a file ───────────────────────────────────────────


class _Box:
    def method(self, a):
        return a, 1.0


def test_uncallable_targets_are_named_precisely_one_sentence_each():
    box = _Box()
    assert "BOUND to a _Box instance" in R._import_problem(box.method, "the t")
    assert "lambda" in R._import_problem(lambda a: (a, 1.0), "the t")

    def nested(a):
        return a, 1.0

    assert "inside another function" in R._import_problem(nested, "the t")
    assert R._import_problem(_fn, "the t") is None


def test_a_main_module_target_is_refused_because_main_moves():
    """It resolves in the emitting process and resolves to something ELSE
    in the reproducer, which is the worst of both."""

    class Fake:
        __module__ = "__main__"
        __qualname__ = "whatever"

        def __call__(self, a):
            return a, 1.0

    assert "__main__" in R._import_problem(Fake(), "the target")


def test_a_name_that_resolves_to_a_different_object_is_refused():
    """A decorator or a monkeypatch. Reproducing against whatever the name
    holds would measure a different program."""

    def impostor(a):
        return a, 2.0

    impostor.__module__ = _fn.__module__
    impostor.__qualname__ = _fn.__qualname__
    problem = R._import_problem(impostor, "the target")
    assert "DIFFERENT object" in problem


# ── the published schema ─────────────────────────────────────────────────────


def test_the_schema_is_versioned_and_its_key_set_is_pinned():
    """SCHEMA is a parsed surface. This test is the thing that makes
    changing it a deliberate act rather than a diff nobody noticed."""
    assert SCHEMA == "stelling.reproducer/1"
    assert SIDECAR_KEYS == (
        "schema", "stelling", "jax", "query", "contract", "verdict",
        "obligation", "relation", "fragment", "equations", "envelope",
        "witness", "execution",
    )
    assert len(set(SIDECAR_KEYS)) == len(SIDECAR_KEYS)


def test_the_execution_modes_are_pinned_too():
    assert EXECUTION_MODES == ("eager", "jit")


# ── the one rule, checked on the artefact ────────────────────────────────────


def test_the_template_itself_never_imports_stelling():
    for lineno, line in enumerate(R._TEMPLATE.splitlines(), 1):
        assert not line.strip().startswith(("import stelling", "from stelling")), (
            f"the emitted file would import stelling at template line {lineno}"
        )


def test_the_emitter_refuses_its_own_output_if_it_ever_imported_stelling(
    monkeypatch,
):
    """Mutation check on the rule the whole feature rests on: poison the
    template with the one line it may never contain, and the emitter must
    refuse rather than write the file."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("z3")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from stelling.preconditions import check

        subject = _nonlinear_subject()
        v = check(
            subject.harness, vacuity_mode="inputs-only", solver_timeout_ms=30_000
        )
        assert v.status == "REFUTED"
        monkeypatch.setattr(
            R, "_TEMPLATE", "import stelling\n" + R._TEMPLATE
        )
        with pytest.raises(ReproducerError, match="checks the tool with the tool"):
            R.write_reproducer(v, subject, "/tmp/stelling-reproduce-never")
    finally:
        jax.config.update("jax_enable_x64", old)


def _nonlinear_subject():
    from reproduce_subjects import product_against_bound

    return Subject(
        name="product",
        fn=product_against_bound,
        relation="<=",
        declarations=(
            ((), "float64", (1.0, 3.0)),
            ((), "float64", (1.0, 3.0)),
        ),
        no_precondition_reason="both factors are free inputs of the caller",
    )


# ── the gate that stops a verdict travelling to the wrong program ────────────


def test_a_verdict_about_another_program_is_refused(tmp_path):
    jax = pytest.importorskip("jax")
    pytest.importorskip("z3")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from stelling.preconditions import check

        subject = _nonlinear_subject()
        v = check(
            subject.harness, vacuity_mode="inputs-only", solver_timeout_ms=30_000
        )
        other = Subject(
            name="product-other-box",
            fn=subject.fn,
            relation="<=",
            declarations=(
                ((), "float64", (1.0, 4.0)),
                ((), "float64", (1.0, 3.0)),
            ),
            no_precondition_reason=subject.no_precondition_reason,
        )
        with pytest.raises(ReproducerError, match="not about this subject"):
            R.write_reproducer(v, other, str(tmp_path))
        # and the honest half of the same gate: the matching subject works
        em = R.write_reproducer(v, subject, str(tmp_path))
        assert em.runnable
    finally:
        jax.config.update("jax_enable_x64", old)


def test_a_verdict_with_no_witness_is_refused_rather_than_half_emitted():
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from stelling.preconditions import check

        subject = _subject(
            fn=_fn, declarations=(((), "float64", (0.0, 1.0)),)
        )
        v = check(subject.harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED"
        with pytest.raises(ReproducerError, match="carries none"):
            R._witness_for(v, None)
    finally:
        jax.config.update("jax_enable_x64", old)


# ── the witness naming contract, read rather than re-derived ─────────────────


def test_witness_value_names_are_parsed_by_the_published_naming():
    s = _subject(declarations=(((), "float64", (0.0, 1.0)),
                               ((2, 3), "float64", (0.0, 1.0))))
    assert R._declaration_of("x0", s) == (0, 0)
    assert R._declaration_of("x1_4", s) == (1, 4)
    with pytest.raises(ReproducerError, match="published declaration naming"):
        R._declaration_of("y0", s)
    with pytest.raises(ReproducerError, match="the subject declares 2"):
        R._declaration_of("x9", s)


def test_don_t_care_elements_are_filled_from_the_box_and_disclosed():
    """Same fill the dispatch layer's replay uses, and the same duty to
    say which elements were invented."""
    s = _subject(declarations=(((3,), "float64", (2.0, 5.0)),))

    class W:
        values = (("x0_1", "3"),)

    values, disclosures = R._point(s, W())
    # the model gave element 1; the other two are filled from the box's lo
    assert values == [["2", "3", "2"]]
    assert len(disclosures) == 2
    assert "not constrained by the model" in disclosures[0]
    assert "x0_0" in disclosures[0] and "x0_2" in disclosures[1]


# ── the side door: the TARGET's module reaching the tool ─────────────────────


def test_a_target_whose_module_imports_stelling_is_disclosed_not_refused():
    """The emitted file never imports stelling — but it imports the module
    the target lives in, and if THAT reaches the tool then the tool is
    loaded after all. Measured the hard way: this feature's own acceptance
    put its targets in the harness module and every reproducer stopped at
    the import line.

    A disclosure, not a refusal: the condition is the user's to fix by
    moving the program out of the harness module, and refusing to emit
    would trade a working file for a lecture.
    """
    # this very module imports stelling at module scope
    leak = R._tool_leak(_fn)
    assert leak is not None
    assert "imports stelling at module scope" in leak
    assert "Move the program out of the harness module" in leak
    # and a program module that does not is silent
    pytest.importorskip("jax")
    from reproduce_subjects import product_against_bound

    assert R._tool_leak(product_against_bound) is None


def test_the_disclosure_reaches_the_emitted_file(tmp_path):
    jax = pytest.importorskip("jax")
    pytest.importorskip("z3")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from stelling.preconditions import check

        leaky = _subject(
            fn=_leaky_target,
            declarations=(
                ((), "float64", (1.0, 3.0)),
                ((), "float64", (1.0, 3.0)),
            ),
        )
        v = check(
            leaky.harness, vacuity_mode="inputs-only", solver_timeout_ms=30_000
        )
        em = R.write_reproducer(v, leaky, str(tmp_path))
        assert "imports stelling at module scope" in em.source
        assert em.runnable  # disclosed, still emitted, still runnable
    finally:
        jax.config.update("jax_enable_x64", old)


def _leaky_target(a, b):
    """Defined in THIS module, which imports stelling — the leak shape."""
    return a * b, 6.0


# ── provenance ───────────────────────────────────────────────────────────────


def test_the_sha_is_a_sha_or_says_why_it_is_not():
    sha = R._stelling_sha()
    assert re.fullmatch(r"[0-9a-f]{40}( \(tree dirty\))?", sha) or sha.startswith(
        "unknown ("
    )


def test_the_fragment_comes_off_the_stamp_and_absence_is_none():
    from stelling.verdict import Stamp, solver_absent

    stamp = Stamp(
        stelling_version="0", jax_version="0", query_content_hash="h",
        arithmetic_mode="a", semantics="s", precision_config="p",
        device_class="d", solver=solver_absent("no solver ran"),
        nonvacuity="n", transfer_tiers=(), transfer_provenance=(),
        assumptions=(), coverage="7 eqns: 7 known (100%)",
    )
    assert R._fragment_of(stamp) is None
    assert R._equation_count(stamp) == 7
