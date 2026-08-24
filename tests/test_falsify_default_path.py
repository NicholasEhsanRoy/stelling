# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""With the flag off, NOTHING changed — and that is measured, not asserted.

The falsification probe ships in 0.2.0, default-off and unaudited — it was
unreleased when this file was written, and the 0.2.0 version bump is what
changed that half. The constraint on it is therefore not "it should be
safe" but "a caller who does not set ``falsify`` must get exactly the
verdict they got before it existed", and a claim of that shape is only
worth what its measurement is worth.

**THREE MEASUREMENTS, BECAUSE "UNCHANGED" HAS THREE MEANINGS.**

*The verdict is unchanged.* :func:`test_the_verdict_is_identical_with_the_flag_absent`
runs a battery through ``check`` twice — once with the keyword absent
entirely, once with it explicitly ``None`` — and compares the whole
:class:`~stelling.verdict.Verdict`, not just its status. Status, stamp,
per-obligation reports, notes and witnesses all have to match, because a
flag that left the status alone and moved a note would still have changed
what a consumer reads.

*The module is never even imported.*
:func:`test_the_probe_module_is_not_imported_on_the_default_path` runs a
default-path ``check`` in a subprocess and asserts ``stelling.falsify`` is
absent from ``sys.modules`` afterwards. This is the measurement that
catches the failure the verdict comparison cannot: an import with a side
effect — a registration, a warning filter, a jax config touch — changes
behaviour without changing any verdict in the battery. It also means a
user who never sets the flag pays nothing for the probe existing.

*The probe cannot run without the keyword.*
:func:`test_no_environment_variable_switches_the_probe_on` pins that the
dial is a keyword and only a keyword. This project reads exactly two
environment variables in ``src/``, neither of which gates analysis, and
the convention is stated in ``preconditions``: opt-ins are keyword
arguments, defaulted off, validated eagerly, so that what a verdict
depended on is visible in the call that produced it. An env var would make
the probe switchable by a CI setting nobody reading the code can see —
which for an instrument that RAISES is a particularly bad property.

**WHY A BATTERY AND NOT ONE HARNESS.** The flag's branch sits on the
VERIFIED path, so a battery of only-VERIFIED harnesses would exercise it
and a battery of only-UNKNOWN harnesses would not exercise it at all. Both
are here, plus a REFUTED and a harness that declines, so the comparison
covers the paths the insertion could have disturbed rather than the one it
sits on.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import textwrap

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

import stelling  # noqa: E402
from stelling.harness import any_array, assert_, assume  # noqa: E402
from stelling.preconditions import check  # noqa: E402

SRC = pathlib.Path(stelling.__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# --- the battery: one harness per outcome the insertion could disturb ----


def verified():
    """VERIFIED: ``x*x >= 0`` over ``[0, 10]``. The path the flag sits on."""
    x = any_array((), "float64", (0.0, 10.0))
    return assert_(x * x >= 0.0)


def verified_with_assume():
    """VERIFIED under a precondition, so the assume path is covered too."""
    x = any_array((), "float64", (0.0, 10.0))
    assume(x >= 2.0)
    return assert_(x >= 1.0)


def unknown():
    """UNKNOWN: interval arithmetic cannot decide ``x*x <= 50`` on
    ``[0, 10]`` — and the early return above the flag's branch is what
    keeps the probe away from it."""
    x = any_array((), "float64", (0.0, 10.0))
    return assert_(jnp.power(x, 2.0) <= 50.0)


def refuted_over_set():
    """Violated over the whole declared set: ``x <= -1`` on ``[0, 10]``."""
    x = any_array((), "float64", (0.0, 10.0))
    return assert_(x <= -1.0)


def array_verified():
    """A VERIFIED with an array declaration and a reduction."""
    d = any_array((4,), "float64", (0.0, 1.0))
    return assert_(jnp.sum(d) <= 4.0)


BATTERY = [
    ("verified", verified),
    ("verified_with_assume", verified_with_assume),
    ("unknown", unknown),
    ("refuted_over_set", refuted_over_set),
    ("array_verified", array_verified),
]


@pytest.mark.parametrize("name,harness", BATTERY, ids=[n for n, _ in BATTERY])
def test_the_verdict_is_identical_with_the_flag_absent(name, harness):
    """The whole verdict, field for field — not just the status."""
    absent = check(harness, vacuity_mode="inputs-only")
    explicit_off = check(harness, vacuity_mode="inputs-only", falsify=None)

    assert absent.status == explicit_off.status
    assert absent.stamp == explicit_off.stamp
    assert absent.obligations == explicit_off.obligations
    assert absent.notes == explicit_off.notes, (
        f"{name}: the notes differ between an absent `falsify` and an "
        f"explicit `falsify=None`. Those two must be the same call."
    )
    assert absent.witnesses == explicit_off.witnesses
    assert absent == explicit_off


@pytest.mark.parametrize("name,harness", BATTERY, ids=[n for n, _ in BATTERY])
def test_the_default_path_carries_no_probe_note(name, harness):
    """Nothing the probe writes may appear on a verdict it did not run on.

    Asserted on the whole battery rather than on the VERIFIED ones,
    because the note is the visible half of the flag and the failure being
    guarded against is a note that leaks onto a path the branch was not
    supposed to reach.
    """
    verdict = check(harness, vacuity_mode="inputs-only")
    for note in verdict.notes:
        assert "falsification probe" not in note, (
            f"{name}: a default-path verdict carries a probe note: {note!r}"
        )


def test_the_flag_ON_changes_only_the_notes_and_only_on_a_VERIFIED():
    """The complement: what turning it on is ALLOWED to change.

    A probe that found nothing may add its work-done line and nothing
    else. If it ever moved the status, the stamp or an obligation report,
    the "can only refute" contract would be broken in the direction that
    matters — the tool would be reporting something it did not establish.
    """
    off = check(verified, vacuity_mode="inputs-only")
    on = check(verified, vacuity_mode="inputs-only", falsify="sample")

    assert on.status == off.status == "VERIFIED"
    assert on.stamp == off.stamp
    assert on.obligations == off.obligations
    assert on.witnesses == off.witnesses
    assert on.notes[: len(off.notes)] == off.notes, (
        "turning the probe on rewrote an existing note; it may only append"
    )
    added = on.notes[len(off.notes) :]
    assert len(added) == 1 and added[0].startswith("falsification probe:")


def test_the_probe_module_is_not_imported_on_the_default_path():
    """THE MEASUREMENT THE VERDICT COMPARISON CANNOT MAKE.

    An import with a side effect changes behaviour without changing any
    verdict in a battery. This runs a default-path ``check`` in a fresh
    interpreter and reads ``sys.modules``.
    """
    script = textwrap.dedent(
        """
        import sys
        import jax
        jax.config.update("jax_enable_x64", True)
        from stelling.harness import any_array, assert_
        from stelling.preconditions import check

        def h():
            x = any_array((), "float64", (0.0, 10.0))
            return assert_(x * x >= 0.0)

        v = check(h, vacuity_mode="inputs-only")
        print("STATUS", v.status)
        print("LOADED", "stelling.falsify" in sys.modules)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(SRC),
            "PATH": "/usr/bin:/bin",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "HOME": "/tmp",
        },
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr[-2000:]}"
    assert "STATUS VERIFIED" in proc.stdout, proc.stdout
    assert "LOADED False" in proc.stdout, (
        f"stelling.falsify was imported by a check() that did not ask for "
        f"it. The default path must not pay for the probe existing, and an "
        f"import is where a side effect would hide.\n{proc.stdout}"
    )


def test_no_environment_variable_switches_the_probe_on():
    """The dial is a keyword and only a keyword.

    Parsed rather than grepped: a substring scan for ``environ`` matches
    the word in a comment, and the first version of this test did exactly
    that and went red on its own module docstring. What is wanted is a
    READ of the environment, which is an AST question.
    """
    probe = ast.parse(
        (SRC / "stelling" / "falsify.py").read_text(encoding="utf-8")
    )
    reads = [
        ast.unparse(n)
        for n in ast.walk(probe)
        if isinstance(n, ast.Attribute)
        and n.attr in ("environ", "getenv", "environb")
    ]
    assert not reads, (
        f"stelling/falsify.py reads the environment ({reads}). This "
        f"project's opt-ins are keyword arguments, defaulted off and "
        f"validated eagerly, so that what a verdict depended on is visible "
        f"in the call that produced it — and this one RAISES, which makes "
        f"an invisible switch worse than usual."
    )
    pre = ast.parse(
        (SRC / "stelling" / "preconditions.py").read_text(encoding="utf-8")
    )
    for node in ast.walk(pre):
        if isinstance(node, ast.Attribute) and node.attr in (
            "environ",
            "getenv",
        ):
            src = ast.unparse(node)
            assert "falsify" not in src, (
                f"the falsify dial is wired to the environment: {src!r}"
            )


def test_the_dial_is_validated_before_anything_is_traced():
    """Eagerly, like every other dial — and for the same measured reason.

    The probe only DOES anything on a VERIFIED, so a typo'd value that
    was checked where it is used would ride green through every UNKNOWN
    path in a project's life and first explode on the day a VERIFIED
    happened. ``preconditions`` records that exact history for
    ``vacuity_mode``.
    """
    for bad in ("yes", "Sample", "sampling", True, 1, 0, ()):
        with pytest.raises(ValueError, match="falsify must be None or"):
            check(unknown, vacuity_mode="inputs-only", falsify=bad)

    # and it raises before the harness is traced at all
    traced = []

    def tattletale():
        traced.append(1)
        x = any_array((), "float64", (0.0, 1.0))
        return assert_(x >= 0.0)

    with pytest.raises(ValueError):
        check(tattletale, vacuity_mode="inputs-only", falsify="nope")
    assert not traced, (
        "the harness was traced before the falsify dial was validated; the "
        "refusal must happen at entry, not after the work"
    )


def test_the_probe_is_handed_the_VERDICT_s_statuses_not_the_propagation_s():
    """THE SOLVER PATH IS WHERE THE PROBE MATTERS MOST, so it must reach it.

    ``propagate`` leaves an obligation ``"unknown"`` and escalation
    upgrades it to ``"discharged"`` in ``make_solver_verdict``. If the
    probe were handed the propagation's view it would decline with "no
    obligation was discharged" on every solver-decided VERIFIED — which is
    precisely the set ``VERIFIED_BARRED_PRIMITIVES`` exists for, and
    precisely where an emission defect that MISSED a violation would sit.

    Measured rather than reasoned: ``x**2 <= 150`` over ``[0, 10]`` is
    true (the max is 100), interval-undecided, and solver-discharged. The
    probe must actually execute points on it.
    """
    pytest.importorskip("z3")

    def solver_verified():
        x = any_array((), "float64", (0.0, 10.0))
        return assert_(jnp.power(x, 2.0) <= 150.0)

    interval_only = check(solver_verified, vacuity_mode="inputs-only")
    assert interval_only.status == "UNKNOWN", (
        "this fixture must be interval-UNDECIDED, or it does not exercise "
        "the solver path the test is about"
    )

    on = check(
        solver_verified,
        vacuity_mode="inputs-only",
        solver_timeout_ms=20_000,
        falsify="sample",
    )
    assert on.status == "VERIFIED"
    line = on.notes[-1]
    assert line.startswith("falsification probe:")
    assert "DECLINED" not in line, (
        f"the probe declined a solver-decided VERIFIED, so it does not "
        f"reach the path it was built for: {line!r}"
    )
    assert " 0 point(s) executed" not in line, (
        f"the probe executed nothing on a solver-decided VERIFIED: {line!r}"
    )
