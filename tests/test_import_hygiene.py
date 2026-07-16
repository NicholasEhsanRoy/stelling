# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Import hygiene: jax only in _jax_compat; private jax modules nowhere."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JAX_IMPORT = re.compile(r"^\s*(import jax\b|from jax\b)")
# built by concatenation so this test file cannot match itself
PRIVATE_JAX = "jax." + "_src"


def test_jax_imported_only_in_compat_module():
    offenders = []
    for path in (REPO / "src").rglob("*.py"):
        if path.name == "_jax_compat.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if JAX_IMPORT.match(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "jax may only be imported in stelling/_jax_compat.py:\n" + "\n".join(offenders)


def test_private_jax_modules_banned_everywhere():
    offenders = []
    for directory in ("src", "tests"):
        for path in (REPO / directory).rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if PRIVATE_JAX in line:
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, f"{PRIVATE_JAX} is banned outright:\n" + "\n".join(offenders)
