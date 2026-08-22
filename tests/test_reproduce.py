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

import dataclasses
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
    assert SCHEMA == "stelling.reproducer/1-provisional"
    assert SIDECAR_KEYS == (
        "schema", "stability", "stelling", "jax", "query", "contract",
        "verdict", "obligation", "relation", "fragment", "equations",
        "envelope", "witness", "witness_filled", "stelling_sha", "x64",
        "execution",
    )
    assert len(set(SIDECAR_KEYS)) == len(SIDECAR_KEYS)


def test_the_schema_is_marked_provisional_in_the_identifier_itself():
    """A reader deciding whether to build on this must be able to tell
    from the artifact, not from a changelog — and the ordinary version
    check must FAIL CLOSED rather than succeed against a guarantee that
    was never given.

    The marking is withdrawn, not redesigned: nothing here has been parsed
    in anger, so no field has been tested by anyone but its author, and
    "small and designed to survive" is a prediction until a consumer has
    tried to live on it. Everything else in this feature is repairable in
    a patch release; this is the one irreversible commitment.
    """
    assert "provisional" in SCHEMA
    # the check a consumer written against a stable 0.1.0 would do
    assert SCHEMA != "stelling.reproducer/1"
    for phrase in (
        "PROVISIONAL / UNSTABLE",
        "added, removed or renamed in any release",
        "without a deprecation cycle",
        "freezes on a CONDITION and not on a version number",
        "parsed real emissions",
    ):
        assert phrase in R.SCHEMA_STABILITY, phrase
    # AND NO RELEASE NUMBER, which is the repair this list is pinning. The
    # string named 0.1.1 as both the release fields could move in and the
    # release it would freeze in; 0.1.1 passed, the running version is
    # 0.2.0.dev0, the schema is still `1-provisional`, and the sentence had
    # become a promise about a release in the past. A commitment stated as a
    # condition cannot expire.
    import re
    assert not re.search(r"\b0\.\d+\.\d+\b", R.SCHEMA_STABILITY), (
        "SCHEMA_STABILITY names a release version again. State the "
        "CONDITION the schema freezes on; a version number goes stale on "
        "the day it ships and this one did."
    )


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


def test_invented_witness_elements_are_named_not_just_disclosed():
    """A consumer must be able to tell an INVENTED value from a solved one,
    and could not: the sidecar published both in `witness` with nothing
    marking which was which, and the disclosure that explained them said
    they were filled "exactly as the replay did", which is false on this
    path — ``solvers._complete_values`` iterates ``sl.inputs``, and an
    element the obligation never reaches is not one of those, so no solver
    and no replay ever assigned it anything."""
    s = _subject(declarations=(((3,), "float64", (2.0, 5.0)),))

    class W:
        values = (("x0_1", "3"),)

    values, filled = R._point(s, W())
    # the model gave element 1; the other two are invented from the box's lo
    assert values == [["2", "3", "2"]]
    assert filled == ["x0_0", "x0_2"]


# ── what the audit found ─────────────────────────────────────────────────────


def test_callable_targets_that_a_file_CAN_name_are_not_reported_uncallable():
    """A file that reports "no execution result" for a target it could have
    executed is a wrongly-silent file, and three importable shapes were
    getting exactly that: a classmethod (the predecessor asked about
    ``__self__``, which a classmethod has), a module-level callable
    INSTANCE (the flax/equinox shape, and the one this module's own docs
    recommend for fixture work), and a ``functools.partial`` (whose
    ``__module__`` is ``functools``). The test is identity — does some
    module-level name resolve to THIS object — not shape."""
    pytest.importorskip("jax")
    import reproduce_subjects as S

    for label, fn, expect in (
        ("classmethod", S.ClassMethodTarget.sides,
         ("reproduce_subjects", "ClassMethodTarget.sides")),
        ("callable instance", S.CALLABLE_INSTANCE,
         ("reproduce_subjects", "CALLABLE_INSTANCE")),
        ("partial", S.PARTIAL_TARGET,
         ("reproduce_subjects", "PARTIAL_TARGET")),
    ):
        module, qualname, problem = R._resolve_target(fn, "the target")
        assert problem is None, (label, problem)
        assert (module, qualname) == expect, (label, module, qualname)
        # and the name really does resolve back to the same object
        import importlib

        obj = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        assert obj is fn or obj.__func__ is fn.__func__, label


def test_an_object_bound_in_a_module_other_than_its_class_is_nameable():
    """An object is not bound where its class is defined. The candidate
    list drew only on the object's own, its type's and its ``func``'s
    ``__module__``, so a callable instance or a partial assembled in one
    module out of parts from another — the shape the fixture guidance
    recommends — reported uncallable. The first round's tests passed only
    because every fixture sat beside its class."""
    pytest.importorskip("jax")
    import reproduce_subjects_bound as B

    assert R._resolve_target(B.BOUND_ELSEWHERE, "t") == (
        "reproduce_subjects_bound", "BOUND_ELSEWHERE", None
    )
    assert R._resolve_target(B.PARTIAL_ELSEWHERE, "t") == (
        "reproduce_subjects_bound", "PARTIAL_ELSEWHERE", None
    )


def test_an_object_no_name_holds_says_so_truthfully():
    """The message claimed the object "carries no __module__/__qualname__"
    for an instance that has both."""
    pytest.importorskip("jax")
    from reproduce_subjects import _CallableInstance

    _, _, problem = R._resolve_target(_CallableInstance(), "the target")
    assert "no importable __module__" not in problem
    assert "NO module-level name anywhere" in problem
    assert "Bind it to one" in problem


def test_the_resolver_is_deterministic_when_several_names_bind_one_object():
    """A GENERATED file must be byte-identical across runs; picking an
    arbitrary alias would make it not."""
    pytest.importorskip("jax")
    import reproduce_subjects as S

    seen = {R._resolve_target(S.ALIASED_TARGET, "t")[:2] for _ in range(5)}
    assert len(seen) == 1
    assert seen.pop()[1] == "ALIASED_TARGET"  # sorted, so the first alias


def test_caller_text_cannot_reach_the_emitted_program_text():
    """A crafted Subject.name produced a file that printed a result line
    and exited 0 WITHOUT EXECUTING ANYTHING. Two mechanisms now: the funnel
    every string passes through, and a refusal at Subject's own door."""
    assert "\n" not in R.one_line("a\nb")
    assert '"""' not in R.one_line('a"""b')
    assert "\\" not in R.one_line("a\\b")
    assert R.one_line("\x00\x1f") == "(empty)"
    with pytest.raises(ReproducerError, match="single physical line"):
        _subject(name='x"""\nprint("== CONFIRMED")\n"""')
    with pytest.raises(ReproducerError, match="triple quote"):
        _subject(name='has """ in it')


def test_the_funnel_holds_at_slots_no_door_guards(tmp_path):
    """``Subject`` guards its own text fields, but three of the six slots
    that reach the emitted docstring are not Subject's: the solver's
    ``produced_by``, the replay sentence and the disclosures all arrive on
    the WITNESS, which accepts any non-empty string. Those are the slots
    the funnel is actually load-bearing for, so they are what this
    measures."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("z3")
    from stelling.verdict import Witness

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from stelling.preconditions import check

        subject = _nonlinear_subject()
        v = check(
            subject.harness, vacuity_mode="inputs-only", solver_timeout_ms=30_000
        )
        real = v.witnesses[0]
        crafted = Witness(
            obligation_index=real.obligation_index,
            values=real.values,
            produced_by='z3"""\nprint("== CONFIRMED")\nimport sys;sys.exit(0)\n"""',
            replay="ok\n== UNREACHABLE",
        )
        # THROUGH THE PUBLIC DOOR. The crafted witness rides on a real
        # verdict, because there is no longer a producer that accepts one
        # from a caller — see the gate test below.
        poisoned = dataclasses.replace(v, witnesses=(crafted,))
        em = R.write_reproducer(poisoned, subject, str(tmp_path))
        text = em.source
        compile(text, "<t>", "exec")            # it still parses
        for line in text.splitlines():
            assert not line.startswith("print("), line
            assert not line.startswith("sys.exit"), line
            assert not line.startswith("== "), line
        # and the crafted text is present, flattened onto its own one line
        assert "z3'''" in text
        assert "ok == UNREACHABLE" in text
    finally:
        jax.config.update("jax_enable_x64", old)


def test_an_unparseable_emission_is_refused_not_written(monkeypatch):
    """The second structural gate on the output. A file that cannot be run
    is not evidence of anything, and a contract name carrying a triple
    quote produced exactly that."""
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
        monkeypatch.setattr(R, "_TEMPLATE", R._TEMPLATE + "\ndef (:\n")
        with pytest.raises(ReproducerError, match="does not parse"):
            R.write_reproducer(v, subject, "/tmp/stelling-reproduce-never")
    finally:
        jax.config.update("jax_enable_x64", old)


def test_no_public_or_private_producer_will_emit_a_cross_program_file(
    tmp_path,
):
    """THE PROPERTY, not the signature.

    Its predecessor asserted that ``closed`` was absent from
    ``write_reproducer``'s parameters and that a substring appeared in its
    source — a test of the harness, named for a property it never touched,
    and the fifth of that shape in this codebase. Meanwhile the scenario
    the name describes still came out through ``reproducer_source``, which
    was public and took ``query_hash``, ``fragment``, ``equations`` and
    ``x64`` straight from its caller: subject B's name and envelope,
    program A's hash, a witness outside B's own envelope, and a CONFIRMED
    from executing B.

    So this runs the scenario, through every function in the module that
    returns or writes a reproducer, and requires each to refuse.
    """
    jax = pytest.importorskip("jax")
    pytest.importorskip("z3")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from stelling.preconditions import check

        a = _nonlinear_subject()
        b = Subject(
            name="program-B",
            fn=a.fn,
            relation="<=",
            declarations=(
                ((), "float64", (0.0, 0.1)),
                ((), "float64", (0.0, 0.1)),
            ),
            no_precondition_reason=a.no_precondition_reason,
        )
        va = check(
            a.harness, vacuity_mode="inputs-only", solver_timeout_ms=30_000
        )
        assert va.status == "REFUTED"

        producers = [
            lambda: R.write_reproducer(va, b, str(tmp_path)),
            lambda: R._reproducer_source(va, b, None),
        ]
        for produce in producers:
            with pytest.raises(ReproducerError, match="not about this subject"):
                produce()

        # and every producer in the module is one of the two above: a third
        # would be a third door, and this is what notices one appearing
        emitting = {
            n for n in dir(R)
            if callable(getattr(R, n))
            and n.endswith(("_source", "reproducer"))
            and not n.startswith("__")
        }
        assert emitting == {"_reproducer_source", "write_reproducer"}, emitting
        assert "reproducer_source" not in R.__all__
        # and the ONE producer takes nothing from its caller but the
        # verdict, the subject and which obligation — the five parameters
        # it used to accept (query hash, fragment, equations, x64, witness)
        # are how the same cross-program file came out after `closed=` was
        # removed from the other door
        import inspect

        params = list(
            inspect.signature(R._reproducer_source).parameters
        )
        assert params == ["verdict", "subject", "obligation_index"], params
    finally:
        jax.config.update("jax_enable_x64", old)


def test_a_witness_element_index_outside_its_declaration_is_refused():
    """`_point` checked the declaration index and not the element index, so
    `x1_77` for a SCALAR declaration was accepted and the value the model
    actually constrained was silently dropped — the file then executed a
    point the solver never produced."""
    s = _subject(declarations=(((), "float64", (0.0, 1.0)),
                               ((2,), "float64", (0.0, 1.0))))
    assert R._declaration_of("x1_1", s) == (1, 1)
    with pytest.raises(ReproducerError, match="names element 77"):
        R._declaration_of("x0_77", s)
    with pytest.raises(ReproducerError, match="names element 5"):
        R._declaration_of("x1_5", s)

    class W:
        values = (("x0_77", "3"),)

    with pytest.raises(ReproducerError, match="names element 77"):
        R._point(s, W())


def test_the_dtype_list_is_the_obligation_layers_and_cannot_drift():
    """A COPY, pinned. The predecessor was a copy that had drifted — it
    omitted float16 and refused it with a reason untrue of float16, so a
    supported declaration could never get a reproducer."""
    from stelling.obligation import _FLOAT_INPUT_DTYPES

    assert set(R._DTYPES) == set(_FLOAT_INPUT_DTYPES)


def test_declared_bounds_go_through_the_declaration_layers_own_classifier():
    """`Fraction(numpy.float32(2.5))` raises a bare TypeError out of the
    emitter, for a bound spelling `ACCEPTED_SPELLINGS` explicitly admits —
    and the fill path called exactly that.

    Every spelling `any_array` accepts must survive both the ENVELOPE and
    the FILL, so both are exercised here on the same values."""
    np = pytest.importorskip("numpy")
    from decimal import Decimal
    from fractions import Fraction as F

    spellings = [np.float32(2.5), np.float64(2.5), Decimal("2.5"), 2.5]
    for raw in spellings:
        assert R._bound(raw) == 2.5, raw
    assert R._bound(5) == 5.0                       # the int spelling too
    assert R._json_number(R._bound(float("inf"))) == "inf"
    assert R._json_number(R._bound(float("-inf"))) == "-inf"

    # THE FILL PATH, on the spelling that used to raise: an element the
    # obligation never reaches is invented from the declared bound, and
    # `Fraction(numpy.float32(...))` is a TypeError.
    for raw in spellings:
        s = _subject(declarations=(((), "float64", (0.0, 1.0)),
                                   ((2,), "float64", (raw, raw))))

        class W:
            values = (("x0", "0"),)

        values, filled = R._point(s, W())
        assert filled == ["x1_0", "x1_1"], raw
        assert [F(t) for t in values[1]] == [F(R._bound(raw))] * 2, raw


def test_a_filename_may_not_leave_the_directory(tmp_path):
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
        for bad in ("../escaped", "sub/dir/x", ".", ".."):
            with pytest.raises(ReproducerError, match="bare file name"):
                R.write_reproducer(v, subject, str(tmp_path), filename=bad)
        em = R.write_reproducer(v, subject, str(tmp_path), filename="fine")
        assert em.path == str(tmp_path / "fine.py")
    finally:
        jax.config.update("jax_enable_x64", old)


def test_a_precision_mismatch_gets_its_own_cause(tmp_path):
    """The program is identical; only the global precision setting moved.
    "this verdict is not about this subject's program" sends a reader to
    look for a program difference that is not there."""
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
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ReproducerError, match="precision setting"):
            R.write_reproducer(v, subject, str(tmp_path))
    finally:
        jax.config.update("jax_enable_x64", old)


def test_a_jax_VERSION_mismatch_gets_its_own_cause(tmp_path):
    """The program is identical and the precision is identical; only the jax
    that traced it moved.

    THE CAUSE THIS COVERS IS MEASURED AND NOT HYPOTHETICAL. jax 0.11.1 added
    an `out_sharding` param to `reduce_max`/`reduce_min`, so a harness
    containing a max or min reduction traces to a different content hash on
    0.11.0 and 0.11.1 with no source line changing — `SOUNDNESS.md`'s
    2026-08-18 entry, whose own text names "a CI job that re-traces and
    diffs" as the consumer that breaks. `_require_same_program` IS that
    consumer, and before the clause this test pins it answered "this verdict
    is not about this subject's program", which sends a reader to look for a
    program difference that is not there.

    DRIVEN ACROSS TWO REAL INTERPRETERS BEFORE IT WAS WRITTEN DOWN — a
    verdict produced on jax 0.11.0 and re-emitted on jax 0.11.1, both at
    x64 on — and reproduced HERE by moving the stamp instead, because this
    suite has one interpreter. What the in-process form checks is the clause
    and its wording; what the two-interpreter run checked is that a real jax
    bump reaches it.
    """
    jax = pytest.importorskip("jax")
    pytest.importorskip("z3")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        import dataclasses

        from stelling.preconditions import check

        subject = _nonlinear_subject()
        v = check(
            subject.harness, vacuity_mode="inputs-only", solver_timeout_ms=30_000
        )
        # the same verdict, stamped by a jax this environment is not running
        # and carrying that jax's (different) query hash
        moved = dataclasses.replace(
            v,
            stamp=dataclasses.replace(
                v.stamp,
                jax_version="0.0.0-not-this-one",
                query_content_hash="f" * 64,
            ),
        )
        with pytest.raises(ReproducerError, match="jax VERSION MOVED") as exc:
            R.write_reproducer(moved, subject, str(tmp_path))
        text = str(exc.value)
        assert "0.0.0-not-this-one" in text, text
        assert jax.__version__ in text, text
        # and it must NOT blame the program, which is the misdiagnosis
        assert "not about this subject's program" not in text, text

        # the control, and it is the half that makes this a measurement: with
        # the stamp's jax version left alone and only the hash moved, the
        # refusal DOES blame the program, because then nothing else differs
        hash_only = dataclasses.replace(
            v, stamp=dataclasses.replace(v.stamp, query_content_hash="f" * 64)
        )
        with pytest.raises(
            ReproducerError, match="not about this subject"
        ) as plain:
            R.write_reproducer(hash_only, subject, str(tmp_path))
        assert "jax VERSION MOVED" not in str(plain.value)
    finally:
        jax.config.update("jax_enable_x64", old)


def _leaky_target(a, b):
    """Defined in THIS module, which imports stelling at module scope — the
    leak shape the emitted file's run-time check must disclose."""
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
        top_despite_coverage=None,
    )
    assert R._fragment_of(stamp) is None
    assert R._equation_count(stamp) == 7
