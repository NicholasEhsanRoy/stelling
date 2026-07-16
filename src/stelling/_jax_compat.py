# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The only module in stelling that imports jax.

Everything jax-shaped is transcribed here, once, into :mod:`stelling.ir`;
every analysis, encoder, and report consumes the IR and never jax. When jax
churns — and ``jax.extend`` explicitly reserves the right to — the blast
radius is this file.

Only public and ``jax.extend`` surfaces are used. Private jax modules are
banned repo-wide (enforced by a pre-commit hook and a test, not a comment).
"""

from __future__ import annotations

import enum
import functools
import warnings

import jax
import jax.extend.core as jex_core
import jax.sharding
import numpy as np

from stelling import ir
from stelling._optional import TESTED_JAX_SERIES, jax_series_tested

__all__ = ["jax_version", "transcribe"]

# jax types that are pure zero-payload sentinels: recording the type name is
# lossless. Matched by name because isinstance would require importing the
# private jax modules they live in. Sentinels with actual content (e.g. a
# real NamedSharding) must NOT be added here — they raise until a rule exists.
_SENTINEL_PARAM_TYPES = frozenset({"UnspecifiedValue"})

# Mesh types are public API; an *empty* mesh (no mesh in scope — the invariant
# state for single-device programs) is zero-payload and safe to record as a
# sentinel. Non-empty meshes mean a sharded program, which stelling does not
# support yet, so they raise.
_MESH_TYPES = tuple(
    t
    for t in (
        getattr(jax.sharding, "Mesh", None),
        getattr(jax.sharding, "AbstractMesh", None),
    )
    if t is not None
)

# (primitive, param) slots that hold transform-time thunks: callables jax
# keeps for its own later transforms, unreachable by construction for any
# analysis stelling performs. Recorded as ir.OpaqueParam (explicitly lossy);
# callables in unlisted slots still raise. Evidence and rationale live in
# design/transparent-primitives.md. Every entry is verified against jax
# 0.10.2 — no speculative entries for other series: on an untested series,
# raising is the correct outcome, surfacing exactly what TESTED_JAX_SERIES
# exists to say.
_OPAQUE_PARAMS = frozenset(
    {
        ("custom_jvp_call", "jvp_jaxpr_fun"),
        ("custom_vjp_call", "fwd_jaxpr_thunk"),
        ("custom_vjp_call", "bwd"),
        ("custom_vjp_call", "out_trees"),
        ("remat2", "policy"),
    }
)


@functools.lru_cache(maxsize=None)  # warn once per distinct version
def _warn_untested_jax(version: str) -> None:
    warnings.warn(
        f"stelling is tested against jax {', '.join(TESTED_JAX_SERIES)}.x but is "
        f"running under jax {version}. Transcription fails loudly on anything it "
        f"does not recognize, and verdicts stamp the exact jax version.",
        RuntimeWarning,
        stacklevel=3,
    )


def jax_version() -> str:
    return jax.__version__


def transcribe(closed_jaxpr) -> ir.ClosedJaxpr:
    """Transcribe a ``jax.extend.core.ClosedJaxpr`` (e.g. the result of
    ``jax.make_jaxpr``) into the jax-free :class:`stelling.ir.ClosedJaxpr`.

    Purely mechanical; raises :class:`stelling.ir.UnsupportedParamError` on
    any eqn param whose type has no transcription rule, naming the primitive
    and the param. Unknown primitives never raise.
    """
    if not jax_series_tested(jax.__version__):
        _warn_untested_jax(jax.__version__)
    return _Transcriber().closed_jaxpr(closed_jaxpr)


class _Transcriber:
    def __init__(self) -> None:
        self._var_ids: dict[int, int] = {}  # id(jax Var) -> IR id, encounter order

    # -- atoms and avals ----------------------------------------------------

    def var(self, v) -> ir.Var:
        vid = self._var_ids.get(id(v))
        if vid is None:
            vid = len(self._var_ids)
            self._var_ids[id(v)] = vid
        return ir.Var(id=vid, aval=self.aval(v.aval))

    def atom(self, a) -> ir.Atom:
        if isinstance(a, jex_core.Literal):
            return ir.Literal(val=self.value(a.val), aval=self.aval(a.aval))
        return self.var(a)

    def aval(self, av) -> ir.Aval:
        dims = []
        for d in getattr(av, "shape", ()):
            if isinstance(d, (int, np.integer)):
                dims.append(int(d))
            else:
                raise ir.TranscriptionError(
                    f"non-static dimension {d!r} in aval {av}; stelling handles "
                    f"fixed shapes only (design commitment 3)"
                )
        dtype = getattr(av, "dtype", None)
        return ir.Aval(
            kind=type(av).__name__,
            shape=tuple(dims),
            dtype=str(dtype) if dtype is not None else None,
            weak_type=bool(getattr(av, "weak_type", False)),
        )

    def value(self, v):
        """A const or Literal value -> IR scalar or inert Array."""
        if v is None or isinstance(v, (bool, int, float, complex, str)):
            return v
        if isinstance(v, np.generic):
            return v.item()
        arr = np.asarray(v)  # materializes jax arrays on host
        return ir.Array(
            dtype=arr.dtype.str,
            shape=tuple(int(d) for d in arr.shape),
            data=arr.tobytes(),
        )

    # -- params ---------------------------------------------------------------

    def param(self, prim: str, name: str, v):
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, enum.Enum):  # before int: IntEnum is an int subclass
            return ir.EnumParam(cls=type(v).__name__, member=v.name)
        if isinstance(v, (int, float, complex)):
            return v
        if isinstance(v, np.dtype):
            return str(v)
        if isinstance(v, type) and issubclass(v, np.generic):  # e.g. np.float32 the class
            return str(np.dtype(v))
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, tuple) and hasattr(v, "_fields"):  # e.g. GatherDimensionNumbers
            return ir.NamedTupleParam(
                cls=type(v).__name__,
                fields=tuple(
                    (f, self.param(prim, f"{name}.{f}", getattr(v, f))) for f in v._fields
                ),
            )
        if isinstance(v, (tuple, list)):
            return tuple(self.param(prim, f"{name}[{i}]", x) for i, x in enumerate(v))
        if isinstance(v, jex_core.ClosedJaxpr):
            return self.closed_jaxpr(v)
        if isinstance(v, jex_core.Jaxpr):
            return self.jaxpr(v)
        if isinstance(v, np.ndarray):
            return self.value(v)
        if (
            type(v).__name__ in _SENTINEL_PARAM_TYPES
            and type(v).__module__.startswith("jax.")
        ):
            return ir.SentinelParam(cls=type(v).__name__)
        if _MESH_TYPES and isinstance(v, _MESH_TYPES):
            if getattr(v, "empty", False):
                return ir.SentinelParam(cls=f"{type(v).__name__}.empty")
            raise ir.UnsupportedParamError(
                f"primitive {prim!r}: param {name!r} is a non-empty mesh "
                f"({v!r}); sharded programs are not supported yet."
            )
        if (prim, name) in _OPAQUE_PARAMS:
            return ir.OpaqueParam(cls=type(v).__qualname__)
        raise ir.UnsupportedParamError(
            f"primitive {prim!r}: param {name!r} has unsupported type "
            f"{type(v).__module__}.{type(v).__qualname__}; refusing to guess — "
            f"dropping a param changes semantics. Add a transcription rule for it."
        )

    # -- structure ------------------------------------------------------------

    def eqn(self, e) -> ir.JaxprEqn:
        prim = e.primitive.name
        return ir.JaxprEqn(
            primitive=prim,
            invars=tuple(self.atom(a) for a in e.invars),
            outvars=tuple(self.var(v) for v in e.outvars),
            params=tuple((k, self.param(prim, k, v)) for k, v in e.params.items()),
            effects=tuple(str(eff) for eff in getattr(e, "effects", ()) or ()),
            source_info=self.source(getattr(e, "source_info", None)),
        )

    def source(self, si) -> tuple[str, ...]:
        if si is None:
            return ()
        try:
            tb = si.traceback
            if tb is None:
                return ()
            return tuple(
                f"{f.file_name}:{f.line_num} ({f.function_name})" for f in tb.frames
            )
        except Exception:
            return (str(si),)

    def debug_info(self, di) -> ir.DebugInfo | None:
        if di is None:
            return None
        return ir.DebugInfo(
            func=str(getattr(di, "func_src_info", "") or ""),
            arg_names=tuple(str(a) for a in (getattr(di, "arg_names", ()) or ())),
            result_paths=tuple(str(r) for r in (getattr(di, "result_paths", ()) or ())),
        )

    def jaxpr(self, j) -> ir.Jaxpr:
        return ir.Jaxpr(
            constvars=tuple(self.var(v) for v in j.constvars),
            invars=tuple(self.var(v) for v in j.invars),
            outvars=tuple(self.atom(a) for a in j.outvars),
            eqns=tuple(self.eqn(e) for e in j.eqns),
            effects=tuple(str(eff) for eff in getattr(j, "effects", ()) or ()),
            debug_info=self.debug_info(getattr(j, "debug_info", None)),
        )

    def closed_jaxpr(self, cj) -> ir.ClosedJaxpr:
        return ir.ClosedJaxpr(
            jaxpr=self.jaxpr(cj.jaxpr),
            consts=tuple(self.value(c) for c in cj.consts),
        )
