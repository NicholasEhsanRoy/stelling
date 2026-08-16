# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""stelling's jax-free IR: a 1:1 mechanical mirror of ``jax.extend.core`` jaxprs.

This module transcribes rather than designs: beyond the commitments named
below, no normalization, no canonical form, no renaming, no derived fields.
If a jaxpr has it, it is mirrored; if a jaxpr doesn't, it is not invented.
Any redesign is Stage 1 work, after the analyses have said what they need.

Two commitments, made deliberately and named as such, plus one hash-scope
decision:

* **The IR is canonical up to alpha-renaming — entailed by hash stability,
  not chosen alongside it.** The commitment is cross-process hash
  stability. A jax ``Var`` carries no serializable identity (historically
  there was ``Var.count``; on current jax the only handle is the object
  itself), so any stable naming must be derived from structure — and
  structure-derived names necessarily collapse alpha-equivalent jaxprs
  (same structure, different tracer-generated identities) into equal IR
  with equal hashes. The collapse is what stability entails, and it is
  also exactly the identity relation verification wants on queries. The
  only free choice left is *which* structure-derived naming —
  first-encounter order over De Bruijn indices, picked for readability,
  with no semantic consequence either way.
* **Hashes are order-independent where jax is orderless — sorting is a
  commitment, not a serialization convenience.** Eqn params (a dict in jax)
  and effects (a set) are stored sorted, so the content hash cannot depend
  on dict/set iteration order.
* **The content hash covers semantics only.** ``source_info`` and
  ``debug_info`` are excluded from :meth:`ClosedJaxpr.content_hash` (and
  kept in :meth:`to_dict`): textually identical programs traced from
  different files must hash alike, or proof caching and the z3-vs-cvc5
  "did both solvers see the same query" check break for no semantic reason.

Nothing here licenses more: stability entails exactly this much
canonicalization and nothing further. The rule for everything else stands:
transcribe.

One dividend of that rule, recorded because it keeps recurring: the mirror
recurses into every sub-jaxpr *regardless of whether anything can analyse
it*. That completeness is what makes the ⊤-coverage denominator
(:mod:`stelling.coverage`) computable at all — an IR that had been clever
and elided subtrees behind unknown primitives, on the grounds that nothing
could look inside them anyway, would report near-total coverage while
having examined almost nothing: the exact confound the instrument exists
to detect. This is the second time *transcribe, don't design* has paid in
a way nobody argued for. The case for it is empirical now, not aesthetic;
stop relitigating it.

**The deserialization door carries a BOUNDED validation pass**
(:func:`ClosedJaxpr.from_dict` — the negative/malformed-shape audit arc
R1/N1/P1 shared one root: ``from_dict`` admitted IR whose avals lie, and
each in-pipeline fix narrowed the exploit by one lie while the class
stayed open). Loaded IR is checked for: shape entries integral and
nonnegative everywhere (avals, ``stelling_any`` shape params,
:class:`Array` value shapes); ``stelling_any`` outvar avals consistent
with their own shape/dtype params; :class:`Array` payload lengths equal
to ``product(shape) x itemsize`` (for dtypes whose itemsize the ``.str``
code names); and aval-vs-value shape consistency for literals and
consts. Violations raise :class:`TranscriptionError` at LOAD — the same
loud posture trace-side transcription has, now symmetric at the
deserialization door. **Explicitly out of scope:** full per-primitive
shape inference (validating every equation's output aval against its
inputs) — that is a shape-inference engine, not a validation pass. The
in-pipeline read gate and shape/decode predicates remain the defence for
lies past this bounded set, and the residual class (adversarial IR with
coordinated lies beyond these checks) is FRAGILE-by-convention:
degrade-not-crash where reachable, never a definite verdict from a
refused value.

This module must never import jax (or anything outside the standard
library): ``stelling._jax_compat`` is the only module allowed to touch jax,
and everything else consumes these types.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field


class TranscriptionError(TypeError):
    """A jaxpr contains something transcription refuses to guess about."""


class UnsupportedParamError(TranscriptionError):
    """An eqn param has a type outside the whitelist.

    Unknown *primitives* are fine (the transfer registry treats them as ⊤
    later); unknown *params* are not — silently dropping one would change
    the semantics of the equation it configures.
    """


@dataclass(frozen=True)
class Aval:
    """Mirror of an abstract value (``ShapedArray`` and friends)."""

    kind: str  # class name of the jax aval, e.g. "ShapedArray"
    shape: tuple[int, ...]
    dtype: str | None  # e.g. "float32"; None for dtype-less avals (tokens)
    weak_type: bool = False

    def __post_init__(self) -> None:
        # local well-formedness at the type level: EVERY construction path
        # (trace, from_dict, internal transforms, direct dataclass
        # construction) passes through here — the construction-path census
        # found the gates lived only at the two doors, leaving direct
        # construction ungated (design/ci-readiness.md, Part A)
        _validate_aval(self, "Aval")


@dataclass(frozen=True)
class Array:
    """An inert ndarray: enough bytes to rehydrate, no numpy needed to hold it."""

    dtype: str  # numpy dtype ``.str`` including byte order, e.g. "<f4"
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        _validate_array_value(self, "Array")  # local, type-level (census)

    def to_numpy(self):
        import numpy as np  # lazy: numpy is not a stelling dependency

        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape).copy()


@dataclass(frozen=True)
class Var:
    id: int  # assigned in first-encounter order at transcription
    aval: Aval


@dataclass(frozen=True)
class Literal:
    val: bool | int | float | complex | str | Array
    aval: Aval

    def __post_init__(self) -> None:
        # the val and the aval are two self-descriptions of one value;
        # their agreement is this type's local invariant (census)
        _validate_value_against_aval(self.val, self.aval, "Literal")


Atom = Var | Literal


@dataclass(frozen=True)
class EnumParam:
    """A param that was an ``enum.Enum`` member (e.g. GatherScatterMode)."""

    cls: str
    member: str


@dataclass(frozen=True)
class NamedTupleParam:
    """A param that was a namedtuple (e.g. GatherDimensionNumbers).

    Field order is the namedtuple's own ``_fields`` order — semantic, so
    never sorted.
    """

    cls: str
    fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class SentinelParam:
    """A param that was a known zero-payload jax sentinel (e.g. the
    ``UnspecifiedValue`` filling ``jit`` sharding params).

    Recording the type name is lossless because these types carry no
    semantic payload; only sentinels explicitly listed in the transcriber
    qualify — anything with actual content still raises.
    """

    cls: str


@dataclass(frozen=True)
class TreeDefParam:
    """A param that was a ``PyTreeDef``: structural plumbing recorded as its
    canonical string form (e.g. ``PyTreeDef({'a': *, 'b': (*, *)})``).

    Mechanically stringified, deliberately not reconstructible — treedefs
    with equal text are treated as identical. The semantic content of the
    primitives that carry them lives in their sub-jaxprs, which are
    transcribed in full.
    """

    text: str


@dataclass(frozen=True)
class OpaqueParam:
    """A param recorded as present but untranscribable, by explicit registry.

    Reserved for transform-time thunks — the callables jax stores in
    ``custom_jvp_call`` / ``custom_vjp_call`` / ``remat`` params for *its
    own* later transforms of the jaxpr. stelling never re-runs jax
    transforms on transcribed IR, so their content is unreachable by
    construction for every analysis stelling performs. Unlike
    :class:`SentinelParam` this is lossy and says so: an analysis whose
    semantics depend on the param's content must refuse when it meets one,
    and the content hash cannot distinguish programs that differ only here
    (see ``design/transparent-primitives.md``). Only (primitive, param)
    pairs listed in the transcriber qualify; unlisted types still raise.
    """

    cls: str


@dataclass(frozen=True)
class JaxprEqn:
    primitive: str
    invars: tuple[Atom, ...]
    outvars: tuple[Var, ...]
    params: tuple[tuple[str, object], ...] = ()
    effects: tuple[str, ...] = ()
    source_info: tuple[str, ...] = ()  # "file:line (function)" frames, as jax recorded them

    def __post_init__(self) -> None:
        # a commitment, not a convenience (see module docstring): the content
        # hash must not depend on dict/set iteration order, so params (a dict
        # in jax) and effects (a set) are stored sorted.
        object.__setattr__(self, "params", tuple(sorted(self.params, key=lambda kv: kv[0])))
        object.__setattr__(self, "effects", tuple(sorted(self.effects)))
        # params MIRRORS A JAX DICT, so its keys are unique by construction on
        # the trace path — but `from_dict` reads a LIST of pairs and a
        # hand-written serialization can repeat a key. Every consumer reads
        # params through `params_dict()`, which silently resolves a repeat
        # LAST-WINS, so a duplicate is not a harmless redundancy: it is two
        # contradictory facts about one param, of which one is picked by
        # serialization order alone.
        #
        # This matters most where key PRESENCE is itself semantic. For
        # `update_jaxpr` on `scatter-add`, an ABSENT key is the hand-built
        # form (the primitive name is the semantic authority) while a key
        # PRESENT with value None is what jax emits and means the REPLACE
        # combiner `lambda x, y: y` — last-wins, NOT accumulation. Measured on
        # this IR before the check: a serialization carrying both
        # ("update_jaxpr", None) and ("update_jaxpr", <the add jaxpr>) loaded
        # without complaint and was admitted by the scatter-add row as
        # accumulate when the add came last, and declined when it came first.
        # Deserialization could not distinguish the two facts because it
        # collapsed them; refusing the duplicate is what keeps them distinct.
        names = [k for k, _ in self.params]
        if len(set(names)) != len(names):
            dups = sorted({n for n in names if names.count(n) > 1})
            _load_check(
                False,
                f"JaxprEqn ({self.primitive!r})",
                f"params carry duplicate key(s) {dups}: params mirrors a jax "
                f"dict, whose keys are unique, and every reader resolves a "
                f"repeat last-wins — so a duplicate silently picks one of two "
                f"contradictory values by serialization order",
            )
        # type-level: a declaration's two self-descriptions must agree
        # (the census's structuralization — see _validate_decl_eqn)
        _validate_decl_eqn(self, "JaxprEqn")

    def params_dict(self) -> dict[str, object]:
        return dict(self.params)


@dataclass(frozen=True)
class DebugInfo:
    func: str = ""
    arg_names: tuple[str, ...] = ()
    result_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Jaxpr:
    constvars: tuple[Var, ...]
    invars: tuple[Var, ...]
    outvars: tuple[Atom, ...]  # outputs may be literals, not only vars
    eqns: tuple[JaxprEqn, ...]
    effects: tuple[str, ...] = ()
    debug_info: DebugInfo | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "effects", tuple(sorted(self.effects)))


@dataclass(frozen=True)
class ClosedJaxpr:
    jaxpr: Jaxpr
    consts: tuple[bool | int | float | complex | str | Array, ...] = field(default=())

    def __post_init__(self) -> None:
        # local pairing invariant: each const against its constvar's aval
        # (component objects self-validated at their own construction)
        for i, (var, val) in enumerate(zip(self.jaxpr.constvars, self.consts)):
            _validate_value_against_aval(
                val, var.aval, f"ClosedJaxpr.consts[{i}] (constvar {var.id})"
            )

    def to_dict(self, *, include_metadata: bool = True) -> dict:
        return _encode(self, include_metadata)

    @staticmethod
    def from_dict(d: dict) -> "ClosedJaxpr":
        obj = _decode(d)
        if not isinstance(obj, ClosedJaxpr):
            raise ValueError(f"expected a serialized ClosedJaxpr, got {type(obj).__name__}")
        _validate_loaded(obj)
        return obj

    def content_hash(self) -> str:
        """sha256 over the canonical serialization of the semantic content.

        Stable across processes and machines (no reliance on builtin
        ``hash``). Excludes ``source_info``/``debug_info`` — see the module
        docstring.
        """
        canonical = json.dumps(
            self.to_dict(include_metadata=False), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- serialization ---------------------------------------------------------
#
# Tagged, recursive, and closed over exactly the types above. ``to_dict`` /
# ``from_dict`` must round-trip losslessly: that is tested, and the content
# hash is defined over this encoding.


def _encode(obj: object, meta: bool) -> object:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, complex):
        return {"k": "complex", "re": obj.real, "im": obj.imag}
    if isinstance(obj, tuple):
        return {"k": "tuple", "items": [_encode(x, meta) for x in obj]}
    if isinstance(obj, Aval):
        return {
            "k": "aval",
            "kind": obj.kind,
            "shape": list(obj.shape),
            "dtype": obj.dtype,
            "weak_type": obj.weak_type,
        }
    if isinstance(obj, Array):
        return {
            "k": "array",
            "dtype": obj.dtype,
            "shape": list(obj.shape),
            "data": base64.b64encode(obj.data).decode("ascii"),
        }
    if isinstance(obj, Var):
        return {"k": "var", "id": obj.id, "aval": _encode(obj.aval, meta)}
    if isinstance(obj, Literal):
        return {"k": "lit", "val": _encode(obj.val, meta), "aval": _encode(obj.aval, meta)}
    if isinstance(obj, EnumParam):
        return {"k": "enum", "cls": obj.cls, "member": obj.member}
    if isinstance(obj, SentinelParam):
        return {"k": "sentinel", "cls": obj.cls}
    if isinstance(obj, OpaqueParam):
        return {"k": "opaque", "cls": obj.cls}
    if isinstance(obj, TreeDefParam):
        return {"k": "treedef", "text": obj.text}
    if isinstance(obj, NamedTupleParam):
        return {
            "k": "ntuple",
            "cls": obj.cls,
            "fields": [[name, _encode(v, meta)] for name, v in obj.fields],
        }
    if isinstance(obj, JaxprEqn):
        d = {
            "k": "eqn",
            "primitive": obj.primitive,
            "invars": [_encode(a, meta) for a in obj.invars],
            "outvars": [_encode(v, meta) for v in obj.outvars],
            "params": [[key, _encode(v, meta)] for key, v in obj.params],
            "effects": list(obj.effects),
        }
        if meta:
            d["source_info"] = list(obj.source_info)
        return d
    if isinstance(obj, DebugInfo):
        return {
            "k": "dbg",
            "func": obj.func,
            "arg_names": list(obj.arg_names),
            "result_paths": list(obj.result_paths),
        }
    if isinstance(obj, Jaxpr):
        d = {
            "k": "jaxpr",
            "constvars": [_encode(v, meta) for v in obj.constvars],
            "invars": [_encode(v, meta) for v in obj.invars],
            "outvars": [_encode(a, meta) for a in obj.outvars],
            "eqns": [_encode(e, meta) for e in obj.eqns],
            "effects": list(obj.effects),
        }
        if meta:
            d["debug_info"] = _encode(obj.debug_info, meta) if obj.debug_info else None
        return d
    if isinstance(obj, ClosedJaxpr):
        return {
            "k": "closed",
            "jaxpr": _encode(obj.jaxpr, meta),
            "consts": [_encode(c, meta) for c in obj.consts],
        }
    raise TypeError(f"stelling.ir cannot encode {type(obj).__name__}")


def _decode(obj: object) -> object:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if not isinstance(obj, dict):
        raise ValueError(f"malformed IR serialization: unexpected {type(obj).__name__}")
    k = obj.get("k")
    if k == "complex":
        return complex(obj["re"], obj["im"])
    if k == "tuple":
        return tuple(_decode(x) for x in obj["items"])
    if k == "aval":
        return Aval(
            kind=obj["kind"],
            shape=tuple(obj["shape"]),
            dtype=obj["dtype"],
            weak_type=obj["weak_type"],
        )
    if k == "array":
        return Array(
            dtype=obj["dtype"],
            shape=tuple(obj["shape"]),
            data=base64.b64decode(obj["data"]),
        )
    if k == "var":
        return Var(id=obj["id"], aval=_decode(obj["aval"]))
    if k == "lit":
        return Literal(val=_decode(obj["val"]), aval=_decode(obj["aval"]))
    if k == "enum":
        return EnumParam(cls=obj["cls"], member=obj["member"])
    if k == "sentinel":
        return SentinelParam(cls=obj["cls"])
    if k == "opaque":
        return OpaqueParam(cls=obj["cls"])
    if k == "treedef":
        return TreeDefParam(text=obj["text"])
    if k == "ntuple":
        return NamedTupleParam(
            cls=obj["cls"],
            fields=tuple((name, _decode(v)) for name, v in obj["fields"]),
        )
    if k == "eqn":
        return JaxprEqn(
            primitive=obj["primitive"],
            invars=tuple(_decode(a) for a in obj["invars"]),
            outvars=tuple(_decode(v) for v in obj["outvars"]),
            params=tuple((key, _decode(v)) for key, v in obj["params"]),
            effects=tuple(obj["effects"]),
            source_info=tuple(obj.get("source_info", ())),
        )
    if k == "dbg":
        return DebugInfo(
            func=obj["func"],
            arg_names=tuple(obj["arg_names"]),
            result_paths=tuple(obj["result_paths"]),
        )
    if k == "jaxpr":
        dbg = obj.get("debug_info")
        return Jaxpr(
            constvars=tuple(_decode(v) for v in obj["constvars"]),
            invars=tuple(_decode(v) for v in obj["invars"]),
            outvars=tuple(_decode(a) for a in obj["outvars"]),
            eqns=tuple(_decode(e) for e in obj["eqns"]),
            effects=tuple(obj["effects"]),
            debug_info=_decode(dbg) if dbg else None,
        )
    if k == "closed":
        return ClosedJaxpr(
            jaxpr=_decode(obj["jaxpr"]),
            consts=tuple(_decode(c) for c in obj["consts"]),
        )
    raise ValueError(f"malformed IR serialization: unknown tag {k!r}")


# --- the bounded from_dict validation pass ----------------------------------
#
# See the module docstring: a fixed list of checks at the deserialization
# door, loud (TranscriptionError), symmetric with trace-side
# transcription; NOT a shape-inference engine. content_hash and the
# encode/decode forms are untouched — validation reads the decoded
# objects and never rewrites them.

import operator as _operator  # noqa: E402  (stdlib; kept local to the pass)


def _safe_repr(obj) -> str:
    """``repr(obj)``, or a visible placeholder when the object refuses.

    For quoting an object this pass is describing in a refusal, and for
    nothing else. `__post_init__` is the public constructor's own body, so
    a `repr` that raises inside a message composed here leaves
    `ir.JaxprEqn(...)` as a raw `RuntimeError` — which is precisely the
    class `_validate_decl_eqn` exists to close, arriving through the
    sentence that was meant to close it (audit 0.2.0 B6 audit 3, F3).
    Measured: a well-formed declaration whose extent has a working
    `__index__` and a refusing `__repr__` raw-crashed the constructor,
    because the message below is an ARGUMENT to `_load_check` and is
    therefore composed whether or not the check fails."""
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 — the message's own totality
        return "<unreadable>"


def _load_extents(shape) -> tuple[str | None, tuple[int, ...]]:
    """A shape's extents NORMALISED to plain ``int``, paired with the
    malformed-extent problem that stopped the normalisation.

    BOUND, NOT DISCARDED — audit 0.2.0 B6 audit 3, F1, which is the same
    finding `interval.dot_general_geometry` and
    `obligation._Slicer._declared_shape` carry. The predicate spelling of
    this function bound `k = _operator.index(d)`, tested `k` and threw it
    away; its callers then re-read the RAW objects — `_validate_array_
    value` with `int(d)` for the byte-length product, `_validate_decl_eqn`
    with `==` against the aval's extents — so what was validated and what
    was used were two different reads of the same self-describing object.
    Returning the normalised extents makes the comparison an `int`-to-
    `int` one that no `__eq__`, `__int__` or second `__index__` can move.

    NOT ONLY `TypeError`: `operator.index` raises whatever `__index__`
    raises, and a `ValueError` or `OverflowError` from an extent left
    `Aval(...)` and `JaxprEqn(...)` raw (audit 0.2.0 B6 audit 3, F2)."""
    out = []
    for d in shape:
        try:
            k = _operator.index(d)
        except Exception:  # noqa: BLE001 — unreadable IS the finding
            return f"non-integer shape extent {_safe_repr(d)}", ()
        if k < 0:
            return f"negative shape extent {k}", ()
        out.append(k)
    return None, tuple(out)


def _load_extent_problem(shape) -> str | None:
    """:func:`_load_extents`' problem alone, for the readers that need the
    verdict and not the count."""
    return _load_extents(shape)[0]


def _load_itemsize(dtype: str) -> int | None:
    """Bytes per element from a numpy ``.str`` dtype code ('<f8' -> 8),
    or None when the code does not name one (the length check is then
    skipped — an honest bound, not a guess)."""
    tail = dtype[2:] if len(dtype) > 2 else ""
    return int(tail) if tail.isdigit() else None


def _load_check(cond: bool, where: str, what: str) -> None:
    if not cond:
        raise TranscriptionError(
            f"malformed IR: {where}: {what} — refused at construction "
            f"(validation runs in the dataclasses' own __post_init__, so "
            f"every path that builds IR — trace, from_dict, or direct "
            f"construction — passes through it; see the module docstring)"
        )


def _validate_array_value(arr: Array, where: str) -> None:
    # the byte-length product is computed from the extents the guard
    # ACTUALLY VALIDATED, not from a second `int(d)` read of the same
    # objects (audit 0.2.0 B6 audit 3, F1)
    problem, extents = _load_extents(arr.shape)
    _load_check(problem is None, where, f"array value has {problem}")
    itemsize = _load_itemsize(arr.dtype)
    if itemsize is not None:
        n = 1
        for d in extents:
            n *= d
        _load_check(
            len(arr.data) == n * itemsize,
            where,
            f"array value of shape {extents} dtype {arr.dtype!r} carries "
            f"{len(arr.data)} byte(s), expected {n * itemsize}",
        )


def _validate_value_against_aval(val, aval: Aval, where: str) -> None:
    if isinstance(val, Array):
        _validate_array_value(val, where)
        _load_check(
            tuple(val.shape) == tuple(aval.shape),
            where,
            f"array value shape {tuple(val.shape)} contradicts the "
            f"recorded aval shape {tuple(aval.shape)}",
        )
    elif isinstance(val, (bool, int, float, complex)):
        _load_check(
            tuple(aval.shape) == (),
            where,
            f"scalar value {val!r} under a non-scalar aval shape "
            f"{tuple(aval.shape)}",
        )
    # str and None values carry no shape claim to cross-check


def _validate_aval(aval: Aval, where: str) -> None:
    problem = _load_extent_problem(aval.shape)
    _load_check(problem is None, where, f"aval has {problem}")


def _validate_atom(atom, where: str) -> None:
    if isinstance(atom, Var):
        _validate_aval(atom.aval, where)
    elif isinstance(atom, Literal):
        _validate_aval(atom.aval, where)
        _validate_value_against_aval(atom.val, atom.aval, where)


def _validate_param_value(v, where: str) -> None:
    if isinstance(v, ClosedJaxpr):
        _validate_closed(v, where)
    elif isinstance(v, Jaxpr):
        _validate_jaxpr(v, where)
    elif isinstance(v, Array):
        _validate_array_value(v, where)
    elif isinstance(v, (tuple, list)):
        # LIST AS WELL AS TUPLE — audit 0.2.0 B6 re-audit, UNSOUND-1. This
        # walk recursed into tuples and not lists, so anything a list param
        # held (a nested ClosedJaxpr, an Array, a shape) reached the rest of
        # the library unvalidated. `_decode` never builds a bare list, so
        # from_dict cannot reach this arm today; it is closed anyway because
        # "the container type I happened to enumerate" is not a reason for a
        # validator to stop looking, and that omission is exactly what let a
        # list `shape` param past `_validate_decl_eqn` above.
        for i, item in enumerate(v):
            _validate_param_value(item, f"{where}[{i}]")
    elif isinstance(v, NamedTupleParam):
        for name, item in v.fields:
            _validate_param_value(item, f"{where}.{name}")


def _validate_jaxpr(jaxpr: Jaxpr, where: str) -> None:
    for i, v in enumerate(jaxpr.constvars):
        _validate_aval(v.aval, f"{where}.constvars[{i}]")
    for i, v in enumerate(jaxpr.invars):
        _validate_aval(v.aval, f"{where}.invars[{i}]")
    for i, a in enumerate(jaxpr.outvars):
        _validate_atom(a, f"{where}.outvars[{i}]")
    for k, eqn in enumerate(jaxpr.eqns):
        w = f"{where}.eqns[{k}] ({eqn.primitive!r})"
        for i, a in enumerate(eqn.invars):
            _validate_atom(a, f"{w}.invars[{i}]")
        for i, v in enumerate(eqn.outvars):
            _validate_aval(v.aval, f"{w}.outvars[{i}]")
        params = dict(eqn.params)
        for name, v in eqn.params:
            _validate_param_value(v, f"{w}.params[{name!r}]")
        _validate_required_params(eqn, w)
        _validate_decl_eqn(eqn, w)


# The param keys jax supplies on EVERY equation for these primitives, measured
# by tracing 42 distinct forms on jax 0.11.0 and checking the key set is
# constant per primitive (`param_key_census.py` in the sweeps repository; the
# census reports which primitives it reached, and a primitive absent from this
# table is simply unconstrained here).
#
# RE-DRIVEN ON THE OTHER TESTED SERIES 2026-08-07, because a required-key
# table measured on one series is a load-path refusal aimed at the other.
# 21 forms on jax 0.10.2 reached 18 of the 19 entries (all but
# `stelling_any`, which is stelling's own and never jax-traced): every
# required key present, and the key set constant per primitive, exactly as
# on 0.11.0. This table is series-stable so far — which is a measurement
# and not a guarantee, so re-drive it per series rather than assuming the
# result travels. `scan` is the standing counterexample to assuming: its
# key set is disjoint across these two series (`num_consts`/`num_carry`
# vs `ft_in`/`ft_out`), and it is not in this table.
#
# WHY THIS EXISTS, and it is the opposite of the defect it closes. Readers were
# taught to test key PRESENCE rather than `.get()`, because a key present with
# value None is a real jax form whose meaning differs from absence — for
# `scatter-add`, present-None is jax's REPLACE combiner while ABSENT is the
# hand-built form, where the primitive name is the semantic authority.
#
# That is right for a traced equation and right for hand-built IR. It is wrong
# for a DESERIALIZED one, because it converts DELETION INTO BLESSING: remove
# `update_jaxpr` from a persisted `.at[k].apply(...)` query and the set-row
# admits it as a plain `.set`. Measured before this check, on operand
# [0, 5, 0, 0] with the key removed: the transfer modelled `element[1] = 2.0`
# where jax computes 5.0 — a box that EXCLUDES the executed value, on the
# interval face, where no solver and no replay is involved to catch it.
#
# The earlier audit concluded "an absent key is structurally unreachable from
# jax", which is true of `bind` and false of `from_dict`. This is the door that
# distinction leaves open, closed at the door rather than at each of the ~30
# readers that would otherwise each have to be right.
_REQUIRED_PARAMS: dict[str, frozenset[str]] = {
    "broadcast_in_dim": frozenset({"broadcast_dimensions", "shape", "sharding"}),
    "concatenate": frozenset({"dimension"}),
    "convert_element_type": frozenset({"new_dtype", "sharding", "weak_type"}),
    "dot_general": frozenset({"dimension_numbers", "out_sharding", "precision",
                              "preferred_element_type"}),
    # MEASURED on jax 0.11.0 and 0.10.2, both series agreeing: a traced
    # `dynamic_slice` carries exactly `slice_sizes`, and a traced
    # `dynamic_update_slice` carries NO params at all — its geometry is the
    # update operand's shape, so there is nothing for it to record and no
    # key a loaded document could be missing. The write row is therefore
    # deliberately ABSENT from this table rather than present with an empty
    # set: absence means "unconstrained", which for a primitive that
    # carries nothing is the true statement.
    "dynamic_slice": frozenset({"slice_sizes"}),
    "exp": frozenset({"accuracy"}),
    "gather": frozenset({"dimension_numbers", "fill_value", "indices_are_sorted",
                         "mode", "slice_sizes", "unique_indices"}),
    "integer_pow": frozenset({"y"}),
    "mul": frozenset({"out_dtype"}),
    "reduce_or": frozenset({"axes"}),
    "reduce_sum": frozenset({"axes", "out_sharding"}),
    "reshape": frozenset({"dimensions", "new_sizes", "sharding"}),
    "scatter": frozenset({"dimension_numbers", "indices_are_sorted", "mode",
                          "unique_indices", "update_consts", "update_jaxpr"}),
    "scatter-add": frozenset({"dimension_numbers", "indices_are_sorted", "mode",
                              "unique_indices", "update_consts", "update_jaxpr"}),
    "slice": frozenset({"limit_indices", "start_indices", "strides"}),
    "sqrt": frozenset({"accuracy"}),
    "squeeze": frozenset({"dimensions"}),
    "stack": frozenset({"axis"}),
    # stelling's own declaration primitive, and the sharpest case of the class.
    # :func:`_validate_decl_eqn` guards the declaration's two self-descriptions
    # against each other — but it reads `params.get("shape")` and skips the
    # comparison when the key is absent, so DELETING `shape` from a persisted
    # declaration silently retires the check. `lo`/`hi` are read with a NaN
    # default, so deleting those declares an unconstrained set while looking
    # like a bound one. This is the primitive the P1 arc's lies all started at.
    "stelling_any": frozenset({"dtype", "hi", "lo", "shape"}),
    "transpose": frozenset({"permutation"}),
}


def _validate_required_params(eqn: "JaxprEqn", where: str) -> None:
    """Refuse a LOADED equation missing a param jax always supplies.

    Load path only, deliberately. Hand-built IR legitimately omits params —
    that is the form whose blessing `update_jaxpr`'s absence carries — and
    `JaxprEqn` stays constructible without them. What is not legitimate is a
    document claiming to be a traced jax program that is missing a key every
    traced instance of that primitive carries.
    """
    required = _REQUIRED_PARAMS.get(eqn.primitive)
    if required is None:
        return
    missing = sorted(required - {k for k, _ in eqn.params})
    _load_check(
        not missing,
        where,
        f"missing param(s) {missing} that jax supplies on every {eqn.primitive!r} "
        f"equation; absence is not a traced form, and readers that test key "
        f"PRESENCE would take it for a hand-built equation and apply the "
        f"primitive's default meaning",
    )


def _validate_decl_eqn(eqn: "JaxprEqn", where: str) -> None:
    """A stelling_any declaration's aval must agree with its OWN
    params — two self-descriptions of one declared set. Called from
    JaxprEqn.__post_init__ (every construction path) and from the
    from_dict walk (kept for its load-path error context).

    THE CHECK IS ON THE EXTENTS, NOT ON THE PARAM'S PYTHON TYPE — audit
    0.2.0 B6 re-audit, UNSOUND-1. It used to run only ``if isinstance(
    shape, tuple)``, so a ``list`` shape param skipped it entirely and a
    declaration saying four elements in its param and two in its aval was
    constructible; `_validate_param_value` recurses into tuples and (until
    the same commit) not into lists, so nothing else read it either. A
    validator that SILENTLY SKIPS a param class it cannot read grants that
    class a pass, which is the opposite of what a validator is for. Any
    sequence of extents is now compared, and a ``shape`` param that is not
    a sequence of extents at all is REFUSED rather than skipped — the two
    self-descriptions cannot be reconciled if one of them cannot be read.

    Bytes and str are sequences and are not shapes: ``tuple("ab")`` is a
    tuple of CHARACTERS, so coercing one would compare something the
    declaration never said. They are refused with the same sentence.

    **This door is not the soundness boundary and closing it does not make
    one.** `ir.py` scopes per-primitive shape inference out of the load
    validation in writing, and `stelling.obligation._Slicer.
    _one_shape_per_value` is what actually stands between a lying document
    and a verdict — including for the form this function deliberately
    still blesses, a declaration with NO ``shape`` param at all (hand-built
    IR legitimately omits params; see `_validate_required_params`). What
    this closes is the constructor's own claim to have compared the two
    self-descriptions when it had not."""
    if eqn.primitive != "stelling_any" or not eqn.outvars:
        return
    params = dict(eqn.params)
    # the declaration's aval must agree with its OWN params —
    # the P1 arc's lies all started at a declaration whose two
    # self-descriptions disagreed
    if "shape" in params:
        shape = params["shape"]
        dims: tuple | None = None
        # A POSITIVE TEST, and the same one the slicer's `_declared_shape`
        # makes — audit 0.2.0 B6 audit 3, the optional item. This read
        # `not isinstance(shape, (str, bytes, bytearray))`, and a
        # `memoryview` walked past it carrying the identical hazard
        # (`tuple(memoryview(b"34"))` is `(51, 52)`). A declaration records
        # its extents in a `tuple` or a `list` — the only forms `_decode`
        # builds and the only forms jax's own params carry — and anything
        # else is refused as unreadable rather than enumerated as banned.
        if isinstance(shape, (tuple, list)):
            try:
                dims = tuple(shape)
            except Exception:  # noqa: BLE001 — unreadable IS the finding
                # not only TypeError, and `isinstance` is a claim about
                # the TYPE and not the object: a `list` SUBCLASS whose
                # `__iter__` raises satisfies the test above. A param
                # whose iteration raises ANYTHING is a self-description
                # that cannot be read, and letting that exception out of
                # `JaxprEqn.__post_init__` raw is the very class this
                # function is closing.
                dims = None
        # EVERY QUOTE HERE IS GUARDED, because a `_load_check` message is
        # an ARGUMENT and is therefore composed on the passing path too: an
        # unguarded `{shape!r}` raw-crashed `JaxprEqn(...)` on a document
        # with nothing wrong with it, whose extent merely had a `__repr__`
        # that refuses (audit 0.2.0 B6 audit 3, F3).
        # the aval side goes through the SAME reader, so the comparison
        # below is int-to-int. `Aval.__post_init__` validates its own
        # extents, so an aval that exists has none of these problems — the
        # read is here because this function may not rest on another
        # guard having run, and it is recorded as a defensive read rather
        # than claimed as a covered one.
        aval_problem, aval_dims = _load_extents(eqn.outvars[0].aval.shape)
        _load_check(
            aval_problem is None,
            where,
            f"stelling_any outvar aval has {aval_problem}",
        )
        _load_check(
            dims is not None,
            where,
            f"stelling_any shape param {_safe_repr(shape)} is not a "
            f"sequence of extents, so it cannot be checked against the "
            f"outvar aval shape {aval_dims} it is the second description "
            f"of",
        )
        problem, param_dims = _load_extents(dims)
        _load_check(
            problem is None, where, f"stelling_any shape param has {problem}"
        )
        # COMPARED AS NORMALISED ints, not as the raw objects — audit 0.2.0
        # B6 audit 3, F1. `dims == tuple(aval.shape)` asked the param's own
        # `__eq__`, so the extents this function had just validated through
        # `__index__` were not the extents it compared.
        _load_check(
            param_dims == aval_dims,
            where,
            f"stelling_any shape param {param_dims} contradicts "
            f"the outvar aval shape {aval_dims}",
        )
    dtype = params.get("dtype")
    if isinstance(dtype, str) and eqn.outvars[0].aval.dtype is not None:
        _load_check(
            dtype == eqn.outvars[0].aval.dtype,
            where,
            f"stelling_any dtype param {dtype!r} contradicts the "
            f"outvar aval dtype {eqn.outvars[0].aval.dtype!r}",
        )


def _validate_closed(closed: ClosedJaxpr, where: str = "query") -> None:
    _validate_jaxpr(closed.jaxpr, where)
    for i, (var, val) in enumerate(zip(closed.jaxpr.constvars, closed.consts)):
        _validate_value_against_aval(
            val, var.aval, f"{where}.consts[{i}] (constvar {var.id})"
        )


def _validate_loaded(closed: ClosedJaxpr) -> None:
    """The bounded from_dict validation pass (module docstring)."""
    _validate_closed(closed)
