# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""stelling.ir contracts — all jax-free."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from stelling import ir

F32 = ir.Aval(kind="ShapedArray", shape=(8,), dtype="float32")


def tiny(primitive: str = "add", source: tuple[str, ...] = ()) -> ir.ClosedJaxpr:
    x = ir.Var(id=0, aval=F32)
    y = ir.Var(id=1, aval=F32)
    two = ir.Literal(val=2.0, aval=ir.Aval(kind="ShapedArray", shape=(), dtype="float32"))
    eqn = ir.JaxprEqn(
        primitive=primitive,
        invars=(x, two),
        outvars=(y,),
        params=(("b", 1), ("a", (1, 2))),
        effects=(),
        source_info=source,
    )
    jaxpr = ir.Jaxpr(constvars=(), invars=(x,), outvars=(y,), eqns=(eqn,))
    return ir.ClosedJaxpr(jaxpr=jaxpr, consts=())


def test_frozen_and_hashable():
    cj = tiny()
    assert {cj: "ok"}[cj] == "ok"
    with pytest.raises(AttributeError):
        cj.jaxpr = None


def test_round_trip_preserves_equality_and_hash():
    cj = tiny(source=("f.py:3 (step)",))
    back = ir.ClosedJaxpr.from_dict(cj.to_dict())
    assert back == cj
    assert back.content_hash() == cj.content_hash()
    assert back.jaxpr.eqns[0].source_info == ("f.py:3 (step)",)


def test_content_hash_distinguishes_semantics():
    assert tiny("add").content_hash() != tiny("mul").content_hash()


def test_content_hash_ignores_source_info():
    a = tiny(source=("here.py:1 (f)",))
    b = tiny(source=("elsewhere.py:99 (g)",))
    assert a != b  # equality still sees metadata
    assert a.content_hash() == b.content_hash()  # the hash does not


def test_params_order_is_stable():
    forward = ir.JaxprEqn(primitive="p", invars=(), outvars=(), params=(("a", 1), ("b", 2)))
    backward = ir.JaxprEqn(primitive="p", invars=(), outvars=(), params=(("b", 2), ("a", 1)))
    assert forward == backward
    assert forward.params_dict() == {"a": 1, "b": 2}


def test_array_round_trip():
    arr = ir.Array(dtype="<f4", shape=(2, 2), data=b"\x00" * 16)
    lit = ir.Literal(val=arr, aval=ir.Aval(kind="ShapedArray", shape=(2, 2), dtype="float32"))
    eqn = ir.JaxprEqn(primitive="p", invars=(lit,), outvars=())
    jaxpr = ir.Jaxpr(constvars=(), invars=(), outvars=(), eqns=(eqn,))
    cj = ir.ClosedJaxpr(jaxpr=jaxpr, consts=(arr, 3.5, True))
    back = ir.ClosedJaxpr.from_dict(cj.to_dict())
    assert back == cj
    assert back.consts[0].data == b"\x00" * 16


def test_param_value_types_round_trip():
    values = (
        None,
        True,
        7,
        2.5,
        complex(1.0, -2.0),
        "text",
        (1, (2, "x")),
        ir.EnumParam(cls="GatherScatterMode", member="PROMISE_IN_BOUNDS"),
        ir.NamedTupleParam(cls="GatherDimensionNumbers", fields=(("offset_dims", (0,)),)),
        ir.SentinelParam(cls="UnspecifiedValue"),
        ir.OpaqueParam(cls="WrappedFun"),
        ir.TreeDefParam(text="PyTreeDef({'a': *, 'b': (*, *)})"),
    )
    eqn = ir.JaxprEqn(
        primitive="p",
        invars=(),
        outvars=(),
        params=tuple((f"k{i}", v) for i, v in enumerate(values)),
    )
    jaxpr = ir.Jaxpr(constvars=(), invars=(), outvars=(), eqns=(eqn,))
    cj = ir.ClosedJaxpr(jaxpr=jaxpr)
    assert ir.ClosedJaxpr.from_dict(cj.to_dict()) == cj


# The repository root, derived from THIS FILE rather than from the working
# directory. The probe below runs `from tests.test_ir import ...` in a
# subprocess, which needs the root importable — and reading that from `cwd`
# made the test pass only when pytest was invoked from the root. It produced a
# spurious red during an overnight run (pytest invoked with an absolute test
# path from a sibling repository), which is the claim-divergence class in a
# TEST rather than a comment: an unstated precondition, with nothing asserting
# it and nothing naming it.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _probe(expr: str) -> str:
    """Run `expr` in a fresh interpreter that can import `tests`, wherever the
    caller's working directory happens to be."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), str(_REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", expr],
        capture_output=True, text=True, check=True, cwd=str(_REPO_ROOT), env=env,
    ).stdout.strip()


def test_content_hash_stable_across_processes():
    """Guards against reliance on salted builtin hash() anywhere."""
    out = _probe(
        "from tests.test_ir import tiny; "
        "print(tiny(source=('a.py:1 (f)',)).content_hash())"
    )
    assert out == tiny(source=("b.py:2 (g)",)).content_hash()


def test_the_cross_process_probe_does_not_depend_on_the_working_directory():
    """PINNED, because the dependency was silent and cost a spurious red.

    An unattended run that hits a spurious red either stops for the wrong
    reason or learns to discount reds, and the second is worse. So the probe's
    independence is asserted rather than assumed: it runs from a directory that
    is not the repository root and must still work.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            out = _probe(
                "from tests.test_ir import tiny; "
                "print(tiny(source=('a.py:1 (f)',)).content_hash())"
            )
        finally:
            os.chdir(cwd)
    assert out == tiny(source=("b.py:2 (g)",)).content_hash()


def test_malformed_serialization_rejected():
    with pytest.raises(ValueError, match="unknown tag"):
        ir.ClosedJaxpr.from_dict({"k": "nonsense"})
    with pytest.raises(ValueError, match="expected a serialized ClosedJaxpr"):
        ir.ClosedJaxpr.from_dict({"k": "var", "id": 0, "aval": {"k": "aval", "kind": "ShapedArray", "shape": [], "dtype": None, "weak_type": False}})
