# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The contract layer's genuinely jax-free tests, in a module with NO
importorskip — so they actually run in a jax-less environment.

Moved verbatim from tests/test_contracts.py (audit round 2, F4): that
module's ``jax = pytest.importorskip("jax")`` is module-level, and a
module-level skip raised during collection skips the ENTIRE file —
measured, the jax-free authoring/sealed-face tests it contained were
reported "1 skipped" in the nojax venv and never ran there. Everything
here imports only :mod:`stelling.contracts` (jax-free by design) plus
stdlib, and must pass in BOTH venvs.
"""

from __future__ import annotations

import dataclasses

import pytest

from stelling import contracts  # jax-free import must always work
from stelling.contracts import (
    Contract,
    ContractVerdict,
    ENSURES_DECLARED,
    EnsuresFace,
)


def test_closed_range_validates_without_jax_or_numpy():
    """Authoring-time bound validation must stay eager in a BARE
    environment: the exact-value classifier it now runs on
    (stelling._bound_spelling) imports numpy lazily and jax never. A
    subprocess with both imports blocked authors, refuses, and returns
    raw values exactly as the full venv does — pinning the laziness
    itself, not just the import graph."""
    import pathlib
    import subprocess
    import sys

    # the src directory THIS test imported contracts from — the subprocess
    # must exercise the same tree, not whatever an editable install points at
    src = pathlib.Path(contracts.__file__).resolve().parents[1]
    code = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("jax", "numpy"):
            raise ImportError(f"{name} blocked for the bare-environment test")

sys.meta_path.insert(0, _Block())
from decimal import Decimal
from stelling.contracts import _closed_range

lo, hi = _closed_range("t", "n", (Decimal("0.1"), 2))
assert lo == Decimal("0.1") and hi == 2 and type(hi) is int
for bad, needle in [
    ((1, 0), "empty envelope"),
    ((0, float("inf")), "non-finite endpoint"),
    ((0, 10**400), "outside binary64's finite range"),
    (("0.1", 1), "not an accepted bound spelling"),
]:
    try:
        _closed_range("t", "n", bad)
    except ValueError as e:
        assert needle in str(e), (bad, str(e))
    else:
        raise AssertionError(f"{bad} was not refused")
assert "numpy" not in sys.modules and "jax" not in sys.modules
print("bare-environment authoring ok")
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": str(src)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "bare-environment authoring ok" in r.stdout
    # and the subprocess really ran THIS tree (an editable install of
    # another checkout raising the same errors would fake a pass)
    probe = subprocess.run(
        [sys.executable, "-c",
         "import stelling.contracts as c; print(c.__file__)"],
        env={"PYTHONPATH": str(src)}, capture_output=True, text=True,
        timeout=60,
    )
    assert str(src) in probe.stdout, probe.stdout


def test_module_imports_without_jax():
    assert contracts.__all__ == [
        "Contract",
        "ContractVerdict",
        "ENSURES_DECLARED",
        "EnsuresFace",
        "check_contract",
        "coefficient_contrast",
        "conditioning_2x2",
        "conditioning_2x2_field",
    ]


# --- the DECLARED-only ensures invariant (jax-free, structural) --------------


def test_ensures_status_token_is_distinct_from_verdict_statuses():
    assert ENSURES_DECLARED == "DECLARED"
    assert ENSURES_DECLARED not in {"VERIFIED", "REFUTED", "UNKNOWN"}


def test_ensures_face_refuses_every_non_declared_status():
    for status in ("VERIFIED", "REFUTED", "UNKNOWN", "declared", "CHECKED", ""):
        with pytest.raises(ValueError, match="DECLARED and nothing else"):
            EnsuresFace(
                statement="s", derivation="d", conditional_on="c",
                status=status,
            )


def test_ensures_face_cannot_be_upgraded_via_replace():
    face = EnsuresFace(statement="s", derivation="d", conditional_on="c")
    assert face.status == ENSURES_DECLARED
    with pytest.raises(ValueError, match="DECLARED and nothing else"):
        dataclasses.replace(face, status="VERIFIED")


def test_ensures_face_requires_populated_texts():
    with pytest.raises(ValueError, match="must be populated"):
        EnsuresFace(statement="", derivation="d", conditional_on="c")


def test_contract_requires_exactly_one_of_ensures_and_reason():
    face = EnsuresFace(statement="s", derivation="d", conditional_on="c")
    with pytest.raises(ValueError, match="exactly one"):
        Contract(name="x", requires_description="r", harness=lambda: (),
                 ensures=None, no_ensures_reason="")
    with pytest.raises(ValueError, match="exactly one"):
        Contract(name="x", requires_description="r", harness=lambda: (),
                 ensures=face, no_ensures_reason="also a reason")


# --- audit F1: the sealed-type funnels ---------------------------------------
# The docstring's DECLARED-only claim was measured false via three public
# routes (audit b5_mutation.py): a subclass with a no-op __post_init__, a
# duck-typed stand-in through Contract, and direct ContractVerdict
# construction. All three must now refuse with the sealed-type wording.


class _UpgradedFace(EnsuresFace):
    def __post_init__(self):  # drop the refusal — the audit's subclass route
        pass


def test_contract_refuses_subclassed_ensures_face():
    laundered = _UpgradedFace(statement="looks checked", derivation="d",
                              conditional_on="c", status="VERIFIED")
    assert laundered.status == "VERIFIED"  # the subclass itself constructs...
    with pytest.raises(ValueError, match="sealed type"):
        Contract(name="x", requires_description="r", harness=lambda: (),
                 ensures=laundered)  # ...but cannot flow anywhere


def test_contract_refuses_duck_typed_ensures():
    from types import SimpleNamespace

    fake = SimpleNamespace(status="VERIFIED", statement="s",
                           derivation="d", conditional_on="c")
    with pytest.raises(ValueError, match="sealed type"):
        Contract(name="x", requires_description="r", harness=lambda: (),
                 ensures=fake)


def test_contract_verdict_refuses_non_sealed_ensures():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="sealed type"):
        ContractVerdict(
            contract_name="x", requires_description="r", requires=object(),
            ensures=SimpleNamespace(status="VERIFIED", statement="s",
                                    derivation="d", conditional_on="c"),
        )
    with pytest.raises(ValueError, match="sealed type"):
        ContractVerdict(
            contract_name="x", requires_description="r", requires=object(),
            ensures=_UpgradedFace(statement="s", derivation="d",
                                  conditional_on="c", status="VERIFIED"),
        )


def test_contract_verdict_requires_exactly_one_of_ensures_and_reason():
    # the pairing check Contract has, now on the directly-constructible
    # verdict container too (audit F1)
    with pytest.raises(ValueError, match="exactly one"):
        ContractVerdict(contract_name="x", requires_description="r",
                        requires=object(), ensures=None, no_ensures_reason="")
    face = EnsuresFace(statement="s", derivation="d", conditional_on="c")
    with pytest.raises(ValueError, match="exactly one"):
        ContractVerdict(contract_name="x", requires_description="r",
                        requires=object(), ensures=face,
                        no_ensures_reason="also a reason")


# --- audit F6: render/stamp integrity of the ensures strings -----------------


def test_ensures_face_refuses_embedded_newlines():
    """A newline in a hand-built face forged column-0 verdict-looking
    lines ('== VERIFIED', a fake solver line) in render and stamp
    (audit b5_mutation.py section 8)."""
    for field, value in (
        ("statement", "ok\n== VERIFIED"),
        ("derivation", "d\rsolver: z3 4.war (fake)"),
        ("conditional_on", "c\ncoverage: fake"),
    ):
        kwargs = dict(statement="s", derivation="d", conditional_on="c")
        kwargs[field] = value
        with pytest.raises(ValueError, match="single physical line"):
            EnsuresFace(**kwargs)


def test_template_ensures_strings_are_single_line():
    face = contracts.conditioning_2x2(
        "float64", (1, 2), (1, 2), (0, 0), 8.0
    ).ensures
    for text in (face.statement, face.derivation, face.conditional_on):
        assert "\n" not in text and "\r" not in text


# --- audit F10: whitespace-only ensures texts --------------------------------


def test_ensures_face_refuses_whitespace_only_texts():
    with pytest.raises(ValueError, match="must be populated"):
        EnsuresFace(statement="   ", derivation="d", conditional_on="c")
    with pytest.raises(ValueError, match="must be populated"):
        EnsuresFace(statement="s", derivation="\t", conditional_on="c")


# --- audit F8: authoring-time refusals for impossible envelopes/shapes -------


def test_authoring_refuses_impossible_ranges():
    """Reversed and NaN ranges declared empty envelopes that authored
    successfully and rendered ('a in [2.0, 1.0]') until first check;
    non-finite endpoints likewise (audit b_emptiness.py B1, h_misc H7).
    Both templates now refuse at authoring, where the guard messages
    already claimed refusal happens."""
    nan = float("nan")
    with pytest.raises(ValueError, match="empty envelope"):
        contracts.conditioning_2x2("float64", (2.0, 1.0), (1, 2), (0, 0), 8.0)
    with pytest.raises(ValueError, match="empty envelope"):
        contracts.conditioning_2x2("float64", (nan, 1.0), (1, 2), (0, 0), 8.0)
    with pytest.raises(ValueError, match="non-finite endpoint"):
        contracts.conditioning_2x2(
            "float64", (float("inf"), float("inf")), (1, 2), (0, 0), 8.0
        )
    with pytest.raises(ValueError, match="empty envelope"):
        contracts.coefficient_contrast((4,), "float64", (2.0, 1.0), 10.0)
    with pytest.raises(ValueError, match="empty envelope"):
        contracts.coefficient_contrast((4,), "float64", (nan, 1.0), 10.0)


def test_authoring_refuses_malformed_shapes():
    """(-1,) and (2,-3) passed the zero-product guard (a product of
    negatives misses them) and ('4',) was silently int-coerced (audit
    c_posings C3/C7); per-dim validation now refuses each, ir.py-style."""
    with pytest.raises(ValueError, match="negative extent"):
        contracts.coefficient_contrast((-1,), "float64", (0.0, 1.0), 10.0)
    with pytest.raises(ValueError, match="negative extent"):
        contracts.coefficient_contrast((2, -3), "float64", (0.0, 1.0), 10.0)
    with pytest.raises(ValueError, match="non-int extent"):
        contracts.coefficient_contrast(("4",), "float64", (0.0, 1.0), 10.0)
    with pytest.raises(ValueError, match="non-int extent"):
        contracts.coefficient_contrast((True,), "float64", (0.0, 1.0), 10.0)


def test_template_authoring_validation_is_eager_and_jax_free():
    # kappa < 1 poses cond_2 <= 1/kappa (f(kappa) = f(1/kappa)), never
    # what the caller asked — refused at authoring time.
    with pytest.raises(ValueError, match="kappa"):
        contracts.conditioning_2x2("float64", (1, 2), (1, 2), (0, 0), 0.5)
    with pytest.raises(ValueError, match="contrast_bound"):
        contracts.coefficient_contrast((4,), "float64", (0.0, 1.0), 0.5)
    with pytest.raises(ValueError, match="zero"):
        contracts.coefficient_contrast((0,), "float64", (0.0, 1.0), 10.0)


def test_check_contract_requires_vacuity_mode():
    c = contracts.conditioning_2x2("float64", (1, 2), (1, 2), (0, 0), 8.0)
    with pytest.raises(TypeError):
        contracts.check_contract(c)  # no silent mode, no silent skip


def test_t3_authoring_refusals_reuse_the_funnels():
    """conditioning_2x2_field: kappa, ranges, shapes — the same authoring
    funnels as the point template and coefficient_contrast, jax-free."""
    field = contracts.conditioning_2x2_field
    ident = lambda t: t  # noqa: E731
    with pytest.raises(ValueError, match="kappa"):
        field((2,), "float64", (0.0, 1.0), 0.5, ident)
    with pytest.raises(ValueError, match="kappa must be a real number"):
        field((2,), "float64", (0.0, 1.0), True, ident)
    with pytest.raises(ValueError, match="empty envelope"):
        field((2,), "float64", (2.0, 1.0), 8.0, ident)
    with pytest.raises(ValueError, match="empty envelope"):
        field((2,), "float64", (float("nan"), 1.0), 8.0, ident)
    with pytest.raises(ValueError, match="non-finite endpoint"):
        field((2,), "float64", (0.0, float("inf")), 8.0, ident)
    with pytest.raises(ValueError, match="negative extent"):
        field((-1,), "float64", (0.0, 1.0), 8.0, ident)
    with pytest.raises(ValueError, match="non-int extent"):
        field(("4",), "float64", (0.0, 1.0), 8.0, ident)
    with pytest.raises(ValueError, match="non-int extent"):
        field((True,), "float64", (0.0, 1.0), 8.0, ident)
    # identity (transform=None) declares theta itself as the family: the
    # trailing (2, 2) requirement is checkable at authoring, so it is
    with pytest.raises(ValueError, match=r"must end in \(2, 2\)"):
        field((4,), "float64", (0.0, 1.0), 8.0, None)
    with pytest.raises(ValueError, match="callable or None"):
        field((2, 2), "float64", (0.0, 1.0), 8.0, "f")
