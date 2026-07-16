# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""``python -m stelling`` — report optional dependencies, smoke-test solvers.

Exit status is nonzero only if an *installed* solver fails its smoke test
(i.e. a broken wheel or binary); missing optional dependencies are
informational.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys

import stelling
from stelling._optional import (
    _OPTIONAL,
    TESTED_JAX_SERIES,
    cvc5_binary,
    jax_series_tested,
    require,
    version,
)


def _selfcheck_z3() -> None:
    z3 = require("z3")
    x = z3.Real("x")
    solver = z3.Solver()
    solver.add(x != x)
    if solver.check() != z3.unsat:
        raise RuntimeError("z3 failed to refute `x != x`")


def _selfcheck_cvc5() -> None:
    cvc5 = require("cvc5")
    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    x = tm.mkConst(tm.getRealSort(), "x")
    solver.assertFormula(tm.mkTerm(cvc5.Kind.DISTINCT, x, x))
    if not solver.checkSat().isUnsat():
        raise RuntimeError("cvc5 failed to refute `x != x`")


_SELFCHECKS = {"z3": _selfcheck_z3, "cvc5": _selfcheck_cvc5}

_SMTLIB_PROBE = "(set-logic QF_LRA)(declare-const x Real)(assert (distinct x x))(check-sat)\n"


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=60, **kw)


def _selfcheck_cvc5_binary(path: str) -> None:
    proc = _run([path, "--lang", "smt2", "-"], input=_SMTLIB_PROBE)
    verdict = proc.stdout.strip()
    if verdict != "unsat":
        raise RuntimeError(
            f"expected `unsat` for `x != x`, got {verdict!r} (stderr: {proc.stderr.strip()!r})"
        )


def _cvc5_binary_version(path: str) -> str:
    # first line is `cvc5 1.3.4 [git ...]` (older builds: `This is cvc5 version 1.3.4`)
    match = re.search(r"^(?:This is )?cvc5(?: version)? (\S+)", _run([path, "--version"]).stdout, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _cvc5_binary_features(path: str) -> list[str]:
    """Optional components compiled into the binary, per --show-config."""
    config = _run([path, "--show-config"]).stdout
    interesting = ("cln", "cocoa", "glpk", "poly", "cryptominisat", "kissat")
    return [f for f in interesting if re.search(rf"^{f}\s*:\s*yes", config, re.IGNORECASE | re.MULTILINE)]


def main() -> int:
    print(f"stelling {stelling.__version__} on Python {platform.python_version()}")
    failures = 0
    for name, spec in _OPTIONAL.items():
        installed = version(name)
        if installed is None:
            line = f'  {name:<9} not installed — pip install "stelling[{spec.extra}]"'
        else:
            line = f"  {name:<9} {installed}"
            if name == "jax" and not jax_series_tested(installed):
                line += f"  [untested series — stelling is tested against jax {', '.join(TESTED_JAX_SERIES)}.x]"
            selfcheck = _SELFCHECKS.get(name)
            if selfcheck is not None:
                try:
                    selfcheck()
                    line += "  [selfcheck: ok]"
                except Exception as e:  # noqa: BLE001 — report broken wheels, don't crash
                    failures += 1
                    line += f"  [selfcheck FAILED: {e}]"
        print(line)

    # External cvc5 binary: the transport for full-featured (e.g. GPL) builds.
    path = cvc5_binary()
    if path is None:
        print(f"  {'cvc5-bin':<9} none — optional; set STELLING_CVC5 or put `cvc5` on PATH")
    else:
        try:
            line = f"  {'cvc5-bin':<9} {_cvc5_binary_version(path)} ({path})"
            _selfcheck_cvc5_binary(path)
            line += "  [selfcheck: ok]"
            features = _cvc5_binary_features(path)
            if features:
                line += f"  [built with: {', '.join(features)}]"
        except Exception as e:  # noqa: BLE001
            failures += 1
            line = f"  {'cvc5-bin':<9} {path}  [selfcheck FAILED: {e}]"
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
