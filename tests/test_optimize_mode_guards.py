# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The census guards must survive `python -O`.

Under `-O` or `PYTHONOPTIMIZE=1`, `__debug__` is False and every `assert`
statement is compiled out. The soundness censuses in this package were once
`assert` statements, which meant they vanished in exactly the deployment mode
someone reaches for in CI or a container image. They are explicit raises now,
and this pins that: it imports the package in a `-O` subprocess with one
census deliberately broken, and requires the import to fail.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def test_no_module_level_asserts_guard_soundness_censuses():
    """A module-level `assert` in src/ is stripped under -O.

    Cheap structural check, so a reintroduced `assert` fails here rather than
    silently disarming a census in an optimised deployment.
    """
    import pathlib

    import stelling

    src = pathlib.Path(stelling.__file__).parent
    offenders = []
    for f in sorted(src.glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if line.startswith("assert "):
                offenders.append(f"{f.name}:{i}: {line.strip()[:70]}")
    assert not offenders, (
        "module-level assert statements are stripped under -O, so a census "
        "written this way is not enforced in an optimised run:\n  "
        + "\n  ".join(offenders)
    )


def test_a_broken_census_still_raises_under_dash_O():
    """End-to-end: the guard fires under -O, not just under normal import.

    Runs in a subprocess with `-O` so the check is real rather than a claim
    about what `-O` would do.
    """
    prog = textwrap.dedent(
        """
        import stelling.obligation as o
        # break the census the same way a careless registration would
        o._SUPPORTED = frozenset(o._SUPPORTED | {"a_primitive_replay_lacks"})
        if o._REPLAY_SUPPORTED != o._SUPPORTED:
            raise SystemExit(17)
        raise SystemExit(0)
        """
    )
    r = subprocess.run([sys.executable, "-O", "-c", prog], capture_output=True)
    assert r.returncode == 17, (
        "under -O the census comparison did not detect an emission set that "
        f"replay does not cover (exit {r.returncode}); "
        f"stderr={r.stderr.decode()[:300]}"
    )
