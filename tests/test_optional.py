# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts: the core imports clean, extras resolve lazily."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stelling import OptionalDependencyError, available, require
from stelling import _optional
from stelling._optional import (
    _OPTIONAL,
    TESTED_JAX_SERIES,
    cvc5_binary,
    jax_series_tested,
    version,
)

OPTIONAL_NAMES = sorted(_OPTIONAL)


def test_import_is_lazy():
    """`import stelling` must not drag in jax or any solver."""
    probe = (
        "import sys, stelling; "
        f"leaked = [m for m in {OPTIONAL_NAMES!r} if m in sys.modules]; "
        "assert not leaked, f'stelling import pulled in {leaked}'"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_unknown_name_rejected():
    with pytest.raises(ValueError, match="unknown optional dependency"):
        available("tensorflow")


def test_require_missing_names_the_extra(monkeypatch):
    monkeypatch.setitem(
        _optional._OPTIONAL,
        "ghost",
        _optional._Optional("stelling_no_such_module", "ghost-dist", "ghost", "testing"),
    )
    with pytest.raises(OptionalDependencyError, match=r"stelling\[ghost\]"):
        require("ghost")


@pytest.mark.parametrize("name", OPTIONAL_NAMES)
def test_require_matches_available(name):
    if available(name):
        module = require(name)
        assert module.__name__ == _OPTIONAL[name].module
        assert version(name) is not None
    else:
        assert version(name) is None
        with pytest.raises(OptionalDependencyError):
            require(name)


def test_cvc5_binary_prefers_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "cvc5"
    fake.touch()
    monkeypatch.setenv("STELLING_CVC5", str(fake))
    assert cvc5_binary() == str(fake)


def test_cvc5_binary_falls_back_to_path(monkeypatch):
    monkeypatch.delenv("STELLING_CVC5", raising=False)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/opt/bin/cvc5" if cmd == "cvc5" else None)
    assert cvc5_binary() == "/opt/bin/cvc5"


def test_cvc5_binary_may_be_absent(monkeypatch):
    monkeypatch.delenv("STELLING_CVC5", raising=False)
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert cvc5_binary() is None


def test_jax_series_tested_matches_series_prefix():
    series = TESTED_JAX_SERIES[0]
    assert jax_series_tested(f"{series}.2")
    assert not jax_series_tested("0.4.38")
    assert not jax_series_tested("99.0.0")
    assert not jax_series_tested("unknown")


def test_tested_series_is_a_hardcoded_literal():
    """The runtime tested-range claim must be its own fact, deliberately not
    derived from the loose [jax] floor in pyproject metadata."""
    tree = ast.parse(Path(_optional.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "TESTED_JAX_SERIES" for target in node.targets
        ):
            assert isinstance(node.value, ast.Tuple), "must be a literal tuple"
            assert all(isinstance(elt, ast.Constant) for elt in node.value.elts)
            return
    pytest.fail("TESTED_JAX_SERIES literal assignment not found in _optional.py")
