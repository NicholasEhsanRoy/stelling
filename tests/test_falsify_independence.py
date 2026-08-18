# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The probe's independence is ENFORCED here, not left to discipline.

A falsification probe is worth exactly what its independence is worth. If
it reaches its answer by consulting the interval propagator or the SMT
encoder, it is a second face asking the same wrong question — and this
repository has already measured that failure mode once, at full cost: an
adversarial audit produced a witness on a trivially true property and
exact-rational replay CONFIRMED it, because both faces drove the same
routing plan (``verdict.Witness``'s docstring, and B6's *"the shared
oracle's ARGUMENTS were not shared"*).

So ``stelling/falsify.py``'s import list is a soundness argument, and an
argument that lives only in a docstring is one an ordinary refactor can
delete without noticing. These tests are the argument's teeth.

**THREE CHECKS, MEASURING THREE DIFFERENT THINGS.**

*The source scan* (:func:`test_the_probe_imports_no_analysis_module`)
parses the module and refuses any import of an analysis module, at any
depth — module scope or inside a function. A function-scope import is the
one this project's other modules use constantly for lazy loading, so a
scan that only read the top of the file would miss the exact shape a
future edit is most likely to take.

*The runtime check*
(:func:`test_the_probe_loads_no_analysis_module_that_TRACING_does_not`)
measures what the scan cannot. A module can reach code without an
``import`` statement — through ``importlib``, through an attribute on an
object handed to it, through a module already in ``sys.modules`` — and
none of that is visible to a parser. It is written as a DIFFERENCE
against tracing the same harness, and its docstring records why the
absolute form was not available: two modules in the deny-list are loaded
by the declaration API itself, before the probe exists.

*The behavioural check* (:func:`test_the_probe_disagrees_with_a_LYING_propagator`)
measures the property the other two are proxies for. It replaces a
transfer function with one that lies, so the analysis discharges an
obligation that is false, and asserts the probe still finds the violation.
An independent probe must be able to CONTRADICT the analysis; one that
merely avoids importing it might still be re-deriving the same answer, and
only this test can tell the difference.

The list of forbidden modules is deliberately the whole analysis surface
rather than the two or three that seem dangerous today. A probe that
imported ``coverage`` would not be obviously unsound, but it would be one
edit away from importing what ``coverage`` imports, and the value of a
bright line is that it does not require judgement at the moment of the
edit.

``stelling._jax_compat`` is the one permitted stelling import, and it is
permitted because it is unavoidable twice over: a program containing the
``stelling_any`` primitive cannot be executed without naming that
primitive, and ``_jax_compat`` is the only module in this package allowed
to import jax at all (``tests/test_import_hygiene.py``). What the probe
uses from it is jax's own evaluation of jax's own primitives, which is
what a user's program does when it runs.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import textwrap

import pytest

import stelling

SRC = pathlib.Path(stelling.__file__).resolve().parent
PROBE = SRC / "falsify.py"

# Everything that participates in deciding a verdict. The probe exists to
# be a second opinion about what these produce, so it may not consult any
# of them. Two of them are nonetheless LOADED before the probe runs -- see
# the runtime test, which is why that test is a difference and not an
# absolute.
FORBIDDEN = frozenset(
    {
        "stelling.affine",
        "stelling.contracts",
        "stelling.coverage",
        "stelling.exactness",
        "stelling.fidelity",
        "stelling.inductive",
        "stelling.interval",
        "stelling.ir",
        "stelling.obligation",
        "stelling.preconditions",
        "stelling.propagate",
        "stelling.reachability",
        "stelling.smt",
        "stelling.solvers",
        "stelling.vacuity",
        "stelling.verdict",
    }
)

ALLOWED_STELLING = frozenset({"stelling._jax_compat", "stelling._optional"})



def _imported_modules(path: pathlib.Path) -> set[str]:
    """Every module name imported anywhere in a file, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import; record it as unresolvable
                names.add("." * node.level + (node.module or ""))
            elif node.module:
                names.add(node.module)
    return names


def test_the_probe_imports_no_analysis_module():
    """THE BRIGHT LINE, at any nesting depth."""
    imported = _imported_modules(PROBE)
    banned = {
        m
        for m in imported
        if m in FORBIDDEN or any(m.startswith(f + ".") for f in FORBIDDEN)
    }
    assert not banned, (
        f"stelling/falsify.py imports {sorted(banned)}. The probe's entire "
        f"value is that it does not reach its answer through the machinery "
        f"it is checking; an import of an analysis module makes it a second "
        f"face asking the same question. If one of these is genuinely "
        f"unavoidable, the module docstring's independence argument has to "
        f"be rewritten to say why — do not just widen this set."
    )


def test_the_only_stelling_imports_are_the_permitted_two():
    """And nothing else from the package sneaks in either.

    Stated as an allow-list rather than only a deny-list: a module added
    to ``stelling`` tomorrow is not in :data:`FORBIDDEN`, and the failure
    mode this file exists to prevent is precisely an import nobody thought
    to forbid.
    """
    imported = _imported_modules(PROBE)
    stelling_imports = {m for m in imported if m == "stelling" or m.startswith("stelling.")}
    unexpected = stelling_imports - ALLOWED_STELLING
    assert not unexpected, (
        f"stelling/falsify.py imports {sorted(unexpected)} from its own "
        f"package. Only {sorted(ALLOWED_STELLING)} are permitted, and the "
        f"reason each is permitted is written in the module docstring."
    )


def test_the_probe_reads_declarations_from_jax_not_from_the_IR():
    """The transcription is not trusted either, and this is why it matters.

    ``stelling.ir`` is in :data:`FORBIDDEN` above, which pins the negative.
    This pins the positive: the probe reads the declared box off a
    ``stelling_any`` equation of jax's own jaxpr. A transcription defect
    that mangled a bound therefore cannot steer the sampler onto the box
    it mis-transcribed — the probe would still sample what the user wrote.
    """
    source = PROBE.read_text(encoding="utf-8")
    assert "jax.make_jaxpr" in source, (
        "the probe no longer traces with jax.make_jaxpr; if it now takes a "
        "traced query from its caller, it has inherited the transcription "
        "and the independence argument must be rewritten"
    )
    assert "stelling_any" in source


def test_the_probe_loads_no_analysis_module_that_TRACING_does_not():
    """The runtime half, as a DIFFERENTIAL — which is the honest shape.

    The first version of this test blocked every analysis module and ran a
    probe in the subprocess. It failed, twice, and both failures were
    facts about the tree rather than about the probe:

    * ``stelling._jax_compat`` -- the probe's one permitted stelling
      import -- imports ``stelling.ir`` at module scope, so ``import
      stelling.harness`` dies before any probe code runs;
    * ``any_array`` itself does ``from stelling.propagate import
      _INT_DTYPE_BOUNDS`` while validating a declaration, so merely
      TRACING a harness loads the propagator.

    Neither is the probe consulting the analysis, and a test that reported
    them as such would be lying in the expensive direction: it would make
    the independence claim look violated when it is not, and the fix would
    be to weaken the test until it passed.

    So the question is asked as a difference instead, which is both true
    and strictly stronger than the absolute form could have been: **does
    running the probe load any analysis module that tracing the same
    harness had not already loaded?** The baseline is measured in the same
    process, on the same harness, immediately before. Anything the probe
    adds is something only the probe wanted, and that is exactly the set
    that must be empty.
    """
    script = textwrap.dedent(
        """
        import sys
        BANNED = %r

        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from stelling.harness import any_array, assert_

        def h():
            x = any_array((), "float64", (0.0, 9.0))
            return assert_(jnp.power(x, 2.0) <= 40.0)

        # BASELINE: what merely tracing the harness pulls in.
        jax.make_jaxpr(h)()
        before = sorted(m for m in BANNED if m in sys.modules)
        print("BASELINE", before)

        from stelling.falsify import probe, VerifiedFalsified
        try:
            r = probe(h, statuses=["discharged"])
            print("EXECUTED", r.points_executed, "DECLINED", r.declined)
        except VerifiedFalsified as e:
            print("FIRED", e.report.points_executed)

        after = sorted(m for m in BANNED if m in sys.modules)
        print("ADDED", sorted(set(after) - set(before)))
        """
        % sorted(FORBIDDEN)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(SRC.parent),
            "PATH": "/usr/bin:/bin",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "HOME": "/tmp",
        },
    )
    assert proc.returncode == 0, (
        f"the probe subprocess failed.\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr[-3000:]}"
    )
    assert "FIRED" in proc.stdout, (
        f"the probe did not find the violation in the subprocess, so this "
        f"test proves nothing about a probe that works.\n{proc.stdout}"
    )
    added = proc.stdout.split("ADDED", 1)[1].strip()
    assert added == "[]", (
        f"running the probe loaded analysis module(s) {added} that tracing "
        f"the same harness had not already loaded. That is the probe "
        f"reaching for the machinery it is checking, whatever the import "
        f"is spelled like -- the source scan above cannot see an "
        f"importlib call or an attribute reach, and this can."
    )


def test_the_probe_disagrees_with_a_LYING_propagator():
    """THE PROPERTY THE OTHER TWO ARE PROXIES FOR.

    Avoiding an import is not the same as being independent. The real
    question is whether the probe can CONTRADICT the analysis, so this
    makes the analysis wrong on purpose -- a transfer for ``pow`` that
    returns the base's own interval, so ``x**2`` over ``[0, 9]`` is
    claimed to be ``[0, 9]`` and ``<= 40`` discharges -- and asserts the
    probe still finds the point where the program violates it.

    With the flag off this is a false VERIFIED and nothing catches it;
    that half is asserted too, because a probe that fires on a query the
    analysis would have refused anyway has demonstrated nothing.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    import stelling.propagate as P
    from stelling.falsify import VerifiedFalsified
    from stelling.harness import any_array, assert_
    from stelling.preconditions import check

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)

    def h():
        x = any_array((), "float64", (0.0, 9.0))
        return assert_(jnp.power(x, 2.0) <= 40.0)  # 9**2 = 81: FALSE

    original = P.TRANSFERS["pow"]
    P.TRANSFERS["pow"] = (lambda eqn, p, ins: [ins[0]], P.TIER_SOUND)
    try:
        lied = check(h, vacuity_mode="inputs-only")
        assert lied.status == "VERIFIED", (
            f"the mutated transfer did not produce a false VERIFIED (got "
            f"{lied.status}), so there is no unsoundness here for the probe "
            f"to disagree with and this test measures nothing"
        )
        with pytest.raises(VerifiedFalsified) as caught:
            check(h, vacuity_mode="inputs-only", falsify="sample")
    finally:
        P.TRANSFERS["pow"] = original
        jax.config.update("jax_enable_x64", old)

    report = caught.value.report
    assert report.falsification is not None
    assert report.falsification.obligation_position == 0
    # and the honest verdict is restored once the lie is removed
    assert check(h, vacuity_mode="inputs-only").status != "VERIFIED"
