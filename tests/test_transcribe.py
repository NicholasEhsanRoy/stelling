# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""jaxpr -> stelling.ir transcription. Skipped wholesale when jax is absent."""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from stelling import _jax_compat, coverage, ir  # noqa: E402


def param_values(cj: ir.ClosedJaxpr):
    """Yield every param value anywhere in the IR, recursing into sub-jaxprs
    and into tuple/namedtuple containers."""
    stack = [cj.jaxpr]
    while stack:
        j = stack.pop()
        for eqn in j.eqns:
            pending = [v for _, v in eqn.params]
            while pending:
                item = pending.pop()
                yield item
                if isinstance(item, ir.ClosedJaxpr):
                    stack.append(item.jaxpr)
                elif isinstance(item, ir.Jaxpr):
                    stack.append(item)
                elif isinstance(item, tuple):
                    pending.extend(item)
                elif isinstance(item, ir.NamedTupleParam):
                    pending.extend(v for _, v in item.fields)


def test_roll_transcribes_and_hashes_deterministically():
    cj = jax.make_jaxpr(lambda x: jnp.roll(x, 1) * 2.0)(jnp.zeros(8))
    first = _jax_compat.transcribe(cj)
    second = _jax_compat.transcribe(cj)
    assert first == second
    assert first.content_hash() == second.content_hash()
    assert first.jaxpr.eqns, "expected a non-empty equation list"
    assert all(isinstance(e.primitive, str) for e in first.jaxpr.eqns)


def test_literals_appear_as_ir_literals():
    cj = jax.make_jaxpr(lambda x: x + 2.0)(jnp.zeros(4))
    out = _jax_compat.transcribe(cj)
    atoms = [a for e in out.jaxpr.eqns for a in e.invars]
    assert any(isinstance(a, ir.Literal) for a in atoms)


def test_consts_become_inert_arrays():
    table = np.arange(6.0, dtype=np.float32).reshape(2, 3)
    cj = jax.make_jaxpr(lambda x: x @ jnp.asarray(table))(jnp.zeros((4, 2)))
    out = _jax_compat.transcribe(cj)
    arrays = [c for c in out.consts if isinstance(c, ir.Array)]
    assert arrays, "expected the closed-over table in consts"
    np.testing.assert_array_equal(arrays[0].to_numpy(), table)


def test_control_flow_recurses_into_params():
    def f(x):
        y = jax.lax.cond(x[0] > 0, lambda v: v + 1.0, lambda v: v - 1.0, x)
        out, ys = jax.lax.scan(lambda c, t: (c + t, c), 0.0, y)
        return out, ys

    out = _jax_compat.transcribe(jax.make_jaxpr(f)(jnp.zeros(4)))
    nested = [
        v for v in param_values(out) if isinstance(v, (ir.ClosedJaxpr, ir.Jaxpr))
    ]
    # two cond branches (tuple-nested) + the scan body (direct param), at least
    assert len(nested) >= 3, "cond branches and scan body should be transcribed sub-jaxprs"


def test_gather_params_survive():
    """gather is the Stage-1 wedge; its namedtuple + enum params must transcribe."""
    out = _jax_compat.transcribe(
        jax.make_jaxpr(lambda x, i: x[i])(jnp.zeros(8), jnp.array([1, 2]))
    )
    values = list(param_values(out))
    assert any(isinstance(v, ir.NamedTupleParam) for v in values)
    assert any(isinstance(v, ir.EnumParam) for v in values)


def test_unknown_param_type_raises_with_names():
    class Weird:
        pass

    transcriber = _jax_compat._Transcriber()
    with pytest.raises(ir.UnsupportedParamError, match=r"'gather'.*'weird'"):
        transcriber.param("gather", "weird", Weird())


def test_bare_object_placeholder_is_a_sentinel():
    """census contact (lineax): identity-only object() placeholders carry no
    payload by construction; subclasses do not qualify."""
    transcriber = _jax_compat._Transcriber()
    assert transcriber.param("linear_solve", "static[0]", object()) == ir.SentinelParam(cls="object")


def test_source_info_is_captured_but_unhashed():
    cj = jax.make_jaxpr(lambda x: x * x)(jnp.zeros(3))
    out = _jax_compat.transcribe(cj)
    assert all(isinstance(e.source_info, tuple) for e in out.jaxpr.eqns)
    d = out.to_dict(include_metadata=False)
    assert "source_info" not in d["jaxpr"]["eqns"][0]


def test_roll_subjaxpr_intact_and_reachable():
    """jnp.roll wraps its real work in a jit eqn (jax 0.10); the sub-jaxpr
    must be transcribed and reachable — see design/transparent-primitives.md."""
    out = _jax_compat.transcribe(jax.make_jaxpr(lambda x: jnp.roll(x, 1))(jnp.zeros(8)))
    nested = [v for v in param_values(out) if isinstance(v, ir.ClosedJaxpr)]
    assert nested, "roll's wrapper eqn should carry a transcribed sub-jaxpr"
    assert any(n.jaxpr.eqns for n in nested), "the sub-jaxpr must arrive intact"


def test_custom_jvp_transcribes_with_opaque_thunk():
    """jax.nn.relu is a custom_jvp function; its derivative thunk becomes an
    explicit OpaqueParam and the primal call_jaxpr stays reachable."""
    out = _jax_compat.transcribe(jax.make_jaxpr(jax.nn.relu)(jnp.zeros(4)))
    values = list(param_values(out))
    assert any(isinstance(v, ir.OpaqueParam) for v in values)
    assert any(isinstance(v, ir.ClosedJaxpr) and v.jaxpr.eqns for v in values)


def test_custom_vjp_transcribes_forward_and_under_grad():
    @jax.custom_vjp
    def f(x):
        return jnp.sin(x)

    def f_fwd(x):
        return f(x), x

    def f_bwd(res, g):
        return (g * jnp.cos(res),)

    f.defvjp(f_fwd, f_bwd)

    fwd = _jax_compat.transcribe(jax.make_jaxpr(f)(jnp.zeros(4)))
    assert sum(isinstance(v, ir.OpaqueParam) for v in param_values(fwd)) >= 3

    grad = _jax_compat.transcribe(
        jax.make_jaxpr(jax.grad(lambda x: f(x).sum()))(jnp.zeros(4))
    )
    # jax AD inlines the bwd math as ordinary eqns (design note): cos appears
    assert "cos" in {e.primitive for e in grad.jaxpr.eqns}


def test_remat_jaxpr_param_transcribes():
    out = _jax_compat.transcribe(
        jax.make_jaxpr(jax.checkpoint(lambda x: jnp.sin(x) * 2.0))(jnp.zeros(4))
    )
    remat = next(e for e in out.jaxpr.eqns if e.primitive == "remat2")
    sub = remat.params_dict()["jaxpr"]
    # jax 0.10 stores an open Jaxpr here; 0.11 merged Jaxpr/ClosedJaxpr into
    # one class, which transcribes as ir.ClosedJaxpr. Either is intact.
    assert isinstance(sub, (ir.Jaxpr, ir.ClosedJaxpr))
    inner = sub if isinstance(sub, ir.Jaxpr) else sub.jaxpr
    assert inner.eqns, "the remat body must arrive intact"


def test_coverage_of_real_trace_is_a_quantity():
    """With an empty registry, everything reached must land in ⊤ — and the
    jit wrapper must count as transparent, its innards as reached."""
    out = _jax_compat.transcribe(jax.make_jaxpr(lambda x: jnp.roll(x, 1))(jnp.zeros(8)))
    cov = coverage.measure(out, known=frozenset())
    assert cov.total > 0
    assert cov.transparent >= 1  # the jit wrapper
    assert cov.unknown > 0  # roll's real work, reached through it
    assert cov.unreached == 0
    assert cov.known == 0 and cov.fraction_known == 0.0
    assert cov.unknown_primitives  # the registry priority signal


def test_treedef_params_stringify():
    import jax.tree_util as jtu

    transcriber = _jax_compat._Transcriber()
    out = transcriber.param("linear_solve", "treedef", jtu.tree_structure({"a": 1, "b": (2, 3)}))
    assert isinstance(out, ir.TreeDefParam)
    assert out.text == "PyTreeDef({'a': *, 'b': (*, *)})"


def test_prng_impl_functions_are_opaque():
    """random_wrap carries jax's key-handling functions; name/tag/key_shape
    survive, the callables become OpaqueParam (census contact, blackjax)."""
    cj = jax.make_jaxpr(
        lambda k: jax.random.normal(jax.random.wrap_key_data(k), (3,))
    )(jnp.zeros((2,), dtype=jnp.uint32))
    out = _jax_compat.transcribe(cj)
    values = list(param_values(out))
    assert any(isinstance(v, ir.OpaqueParam) for v in values)
    assert any(
        isinstance(v, ir.NamedTupleParam) and v.cls == "PRNGImpl" for v in values
    )


def test_untested_jax_series_warns(monkeypatch):
    _jax_compat._warn_untested_jax.cache_clear()  # warn-once must not hide the fence
    cj = jax.make_jaxpr(lambda x: x + 1.0)(jnp.zeros(2))
    monkeypatch.setattr(jax, "__version__", "0.99.0")
    with pytest.warns(RuntimeWarning, match="0.99.0"):
        _jax_compat.transcribe(cj)


def test_tested_jax_series_is_silent():
    """Fails when CI's jax outruns TESTED_JAX_SERIES — bump it deliberately.

    cache_clear matters: the runtime warning is once-per-version, so without
    it this test passes whenever any earlier test already consumed the
    warning — a test-order-dependent fence, discovered live on the 0.11 bump.
    """
    _jax_compat._warn_untested_jax.cache_clear()
    cj = jax.make_jaxpr(lambda x: x + 1.0)(jnp.zeros(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _jax_compat.transcribe(cj)


def test_content_hash_stable_across_processes():
    probe = (
        "import jax, jax.numpy as jnp\n"
        "from stelling import _jax_compat\n"
        "cj = jax.make_jaxpr(lambda x: jnp.roll(x, 1) * 2.0)(jnp.zeros(8))\n"
        "print(_jax_compat.transcribe(cj).content_hash())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    local = _jax_compat.transcribe(
        jax.make_jaxpr(lambda x: jnp.roll(x, 1) * 2.0)(jnp.zeros(8))
    ).content_hash()
    assert out == local
