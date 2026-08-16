# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""stelling's jax-free IR: a 1:1 mechanical mirror of ``jax.extend.core`` jaxprs.

This module transcribes rather than designs: beyond the commitments named
below, no normalization, no canonical form, no renaming, no derived fields.
If a jaxpr has it, it is mirrored; if a jaxpr doesn't, it is not invented.
Any redesign is Stage 1 work, after the analyses have said what they need.

**WHAT THIS MODULE DOES TO A DOCUMENT IT IS GIVEN IS WRITTEN ONCE, IN
:data:`CANONICALIZATIONS`, AND IS NOT RESTATED HERE.** This paragraph used
to be the list, introduced as *"two commitments … plus one hash-scope
decision"* and closed with *"Nothing here licenses more: stability entails
exactly this much canonicalization and nothing further"* — a counted,
hand-maintained enumeration in prose. It said three where the code did
four: `f729d70` had added the extents install and nothing added it to the
sentence, and this commit adds a fifth (audit 0.2.0 B6 audit 6, F4; it is
the eighth instance in this batch of a claim that stopped matching the
code beside it, and the previous two were inside the fix meant to close
the seventh).

:data:`CANONICALIZATIONS` is data rather than prose, and each entry is
DEMONSTRATED by a witness in `tests/test_ir_canonicalization.py` that
collapses two spellings of one document — so an entry that stops
describing the code reds, an entry with no witness reds, a witness with
no entry reds, and a docstring that restates an entry reds. **What no
test here can catch is a canonicalization added with no entry at all**,
which is why the list is a short structure a reader can diff against a
commit rather than a paragraph they have to re-derive.

Stability is what licenses these and nothing else licenses more. The rule
for everything the list does not name stands: transcribe.

The one thing on that list that is a *consequence* rather than an act of
this module, recorded here because it is what the others are for: **the IR
is canonical up to alpha-renaming — entailed by hash stability, not chosen
alongside it.** The commitment is cross-process hash stability. A jax
``Var`` carries no serializable identity (historically there was
``Var.count``; on current jax the only handle is the object itself), so any
stable naming must be derived from structure — and structure-derived names
necessarily collapse alpha-equivalent jaxprs (same structure, different
tracer-generated identities) into equal IR with equal hashes. The collapse
is what stability entails, and it is also exactly the identity relation
verification wants on queries. The only free choice left is *which*
structure-derived naming — first-encounter order over De Bruijn indices,
picked for readability, with no semantic consequence either way. The ids
themselves are minted by :mod:`stelling._jax_compat` at transcription;
this module transcribes whatever ids it is handed.

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


# EVERY CANONICALIZATION THIS MODULE MAKES, WRITTEN ONCE — audit 0.2.0 B6
# audit 6, F4. A ``(name, what)`` pair per entry, deliberately plain
# tuples rather than a dataclass, because :data:`_CANONICAL_IR_TYPES` is
# computed from this module's dataclasses and a commitment is not a thing
# a document may carry.
#
# The module docstring cites this object and does not restate it; the
# witnesses live in `tests/test_ir_canonicalization.py`, one per entry,
# each collapsing two spellings of one document — so an entry that stops
# describing the code reds, an entry with no witness reds, and a witness
# with no entry reds. The mechanism is that test; this comment is not.
CANONICALIZATIONS: tuple[tuple[str, str], ...] = (
    (
        "alpha-renaming",
        "A CONSEQUENCE, not an act of this module: `Var` identity is a "
        "structure-derived id minted at transcription, so alpha-equivalent "
        "jaxprs are equal IR with equal hashes. This module transcribes "
        "whatever ids it is handed.",
    ),
    (
        "param and effect order",
        "`JaxprEqn.params` (a dict in jax) and `JaxprEqn.effects` / "
        "`Jaxpr.effects` (a set in jax) are stored SORTED, so the content "
        "hash cannot depend on dict or set iteration order.",
    ),
    (
        "hash scope",
        "`source_info` and `debug_info` are excluded from "
        "`ClosedJaxpr.content_hash` and kept in `to_dict`: textually "
        "identical programs traced from different files must hash alike, "
        "or proof caching and the z3-vs-cvc5 'did both solvers see the "
        "same query' check break for no semantic reason.",
    ),
    (
        "shape extents",
        "An `Aval`'s, an `Array`'s and a `stelling_any` declaration's "
        "extents are stored as plain `int` in a plain `tuple` — the value "
        "the guard read through `__index__` and compared, INSTALLED, so a "
        "later reader cannot be handed a second answer. `True` therefore "
        "stores as `1`; `operator.index(True)` is `1` exactly, and no jax "
        "trace produces a boolean extent.",
    ),
    (
        "stored value types",
        "Every document-supplied value in this module's dataclasses is "
        "stored as an EXACT instance of a type the module is closed over — "
        "a subclass is read once through its base type's own accessor and "
        "replaced by the exact twin, a `list` is stored as a `tuple`, and "
        "a type with no exact form to store is refused. See the "
        "canonicalization door.",
    ),
)


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
        #
        # AND THE GUARD'S OWN VALUE IS INSTALLED, not discarded — audit
        # 0.2.0 B6 audit 5, F1. `_validate_aval` normalises every extent
        # through `__index__`; storing what it returned is what makes the
        # extents a LATER reader sees the extents this check passed on.
        # See :class:`JaxprEqn`'s `__post_init__` for the finding this is
        # the sibling of.
        #
        # CANONICALIZED FIRST, so what is validated is what is stored — a
        # guard that decides about a value the object is free to replace on
        # the next read has decided about something else (audit 0.2.0 B6
        # audit 6; see the canonicalization door). `kind` and `dtype` are
        # dispatch keys all over the library and are document-supplied
        # exactly as `shape` is.
        #
        # `shape` IS SKIPPED BECAUSE IT HAS ITS OWN RULE, and that rule is
        # the stronger one: `_load_extents` reads every extent ONCE through
        # `operator.index` and the line below installs what it returned, so
        # an aval extent may be any object with a working `__index__` — a
        # `numpy` integer, or one of this suite's `_TwoFacedExtent`s — and
        # is stored as a plain `int` either way. The generic door would
        # refuse those for their TYPE, which is a narrowing nothing asked
        # for, and would refuse them BEFORE the rule that already
        # canonicalizes them.
        _canonicalise(self, "Aval", skip=("shape",))
        object.__setattr__(self, "shape", _validate_aval(self, "Aval"))


@dataclass(frozen=True)
class Array:
    """An inert ndarray: enough bytes to rehydrate, no numpy needed to hold it."""

    dtype: str  # numpy dtype ``.str`` including byte order, e.g. "<f4"
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        # local, type-level (census) — and the validated extents are
        # INSTALLED, so `_encode`, `to_numpy` and the emission read what
        # the byte-length check was computed from (audit 0.2.0 B6 audit 5,
        # F1). Measured before this line: an `Array` whose `shape` was a
        # `tuple` subclass answering `(2,)` once and `(1,)` afterwards
        # passed the length check at two elements and encoded as one.
        #
        # And canonicalized first: `data` is document-supplied bytes whose
        # `__len__` the byte-length check below reads, and a `bytes`
        # SUBCLASS can answer that twice (audit 0.2.0 B6 audit 6).
        # `shape` is skipped for the reason `Aval` gives — the extent rule
        # is `_load_extents`, it is stronger than a type test, and it
        # already installs what it read.
        _canonicalise(self, "Array", skip=("shape",))
        object.__setattr__(self, "shape", _validate_array_value(self, "Array"))

    def to_numpy(self):
        import numpy as np  # lazy: numpy is not a stelling dependency

        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape).copy()


@dataclass(frozen=True)
class Var:
    id: int  # assigned in first-encounter order at transcription
    aval: Aval

    def __post_init__(self) -> None:
        # `id` IS this value's identity: the slicer, the propagator's
        # environment and the emission all key on it, by `==` and by
        # `hash`, and an `int` SUBCLASS answers those two independently.
        # `aval` must be an `Aval` and not a subclass, because a subclass
        # may make any of its fields a property (audit 0.2.0 B6 audit 6).
        _canonicalise(self, "Var")


@dataclass(frozen=True)
class Literal:
    val: bool | int | float | complex | str | Array
    aval: Aval

    def __post_init__(self) -> None:
        # the val and the aval are two self-descriptions of one value;
        # their agreement is this type's local invariant (census) — checked
        # on the canonical value, so the emission reads the number this
        # comparison was made about (audit 0.2.0 B6 audit 6)
        _canonicalise(self, "Literal")
        _validate_value_against_aval(self.val, self.aval, "Literal")


Atom = Var | Literal


@dataclass(frozen=True)
class EnumParam:
    """A param that was an ``enum.Enum`` member (e.g. GatherScatterMode)."""

    cls: str
    member: str

    def __post_init__(self) -> None:
        # `member` is READ AS A NAME and dispatched on —
        # `propagate._recognised_precision` takes `prec.member.upper()`,
        # and compares `cls` with `== "Precision"` on the same line — so a
        # `str` subclass can make `upper()` and the text it carries
        # disagree (audit 0.2.0 B6 audit 6)
        _canonicalise(self, "EnumParam")


@dataclass(frozen=True)
class NamedTupleParam:
    """A param that was a namedtuple (e.g. GatherDimensionNumbers).

    Field order is the namedtuple's own ``_fields`` order — semantic, so
    never sorted.
    """

    cls: str
    fields: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        # a namedtuple param's field VALUES are the dimension numbers a
        # transfer and an emission both read (audit 0.2.0 B6 audit 6)
        _canonicalise(self, "NamedTupleParam")


@dataclass(frozen=True)
class SentinelParam:
    """A param that was a known zero-payload jax sentinel (e.g. the
    ``UnspecifiedValue`` filling ``jit`` sharding params).

    Recording the type name is lossless because these types carry no
    semantic payload; only sentinels explicitly listed in the transcriber
    qualify — anything with actual content still raises.
    """

    cls: str

    def __post_init__(self) -> None:
        # `cls` is the whole payload and is compared by name
        # (audit 0.2.0 B6 audit 6)
        _canonicalise(self, "SentinelParam")


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

    def __post_init__(self) -> None:
        # "treedefs with equal text are treated as identical" is a claim
        # about `text`, and a `str` subclass can make equal text compare
        # unequal (audit 0.2.0 B6 audit 6)
        _canonicalise(self, "TreeDefParam")


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

    def __post_init__(self) -> None:
        # audit 0.2.0 B6 audit 6
        _canonicalise(self, "OpaqueParam")


@dataclass(frozen=True)
class JaxprEqn:
    primitive: str
    invars: tuple[Atom, ...]
    outvars: tuple[Var, ...]
    params: tuple[tuple[str, object], ...] = ()
    effects: tuple[str, ...] = ()
    source_info: tuple[str, ...] = ()  # "file:line (function)" frames, as jax recorded them

    def __post_init__(self) -> None:
        # EVERYTHING BUT THE PARAM VALUES IS CANONICALIZED FIRST — audit
        # 0.2.0 B6 audit 6, and the reason is the very next statement.
        # `primitive` is the transfer registry's dispatch key, read by `==`
        # and by `hash`; `invars`/`outvars` carry the `Var.id`s the
        # environment is keyed on; `effects` is sorted below and
        # `source_info` is quoted into every report. The param VALUES are
        # held back for `_canonical_param_values` at the foot of this
        # method, because `_validate_decl_eqn` owns two of them and its
        # container rule is the sentence a `shape` param should be refused
        # by.
        _canonicalise(self, "JaxprEqn", skip=("params",))
        object.__setattr__(
            self, "params", _canonical_param_keys(self.params, "JaxprEqn")
        )
        # a commitment, not a convenience (see module docstring): the content
        # hash must not depend on dict/set iteration order, so params (a dict
        # in jax) and effects (a set) are stored sorted.
        #
        # THE SORT IS A READ OF THE KEYS, which is why they are canonical
        # before it. Measured on `f729d70`: three honest params
        # `aaa`/`bbb`/`ccc` whose FIRST key is a `str` subclass returning
        # True from `__gt__` stored as `bbb, ccc, aaa` — an order `sorted`
        # never puts them in, decided by the key's opinion of itself rather
        # than by the key. (`__lt__` alone is not enough and that is worth
        # knowing: a subclass's REFLECTED operation is tried first, so
        # `"aaa" < Sub("bbb")` reaches `Sub.__gt__`, and a subclass that
        # overrides only `__lt__` is never asked.) The commitment this line
        # states is exactly that the stored order depends on nothing but
        # the keys.
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
        #
        # THIS CHECK ASKS TWO DIFFERENT PROTOCOLS OF THE SAME KEYS, and it
        # is sound because `_canonical_param_keys` ran: `set(names)` is
        # HASH-based and `names.count(n)` is EQ-based, so before the keys
        # were exact `str`, two subclasses with equal text and different
        # `__hash__` were two set elements AND two count hits — no
        # duplicate seen, and a document carrying both
        # ("update_jaxpr", None) and ("update_jaxpr", <the add jaxpr>) was
        # ACCEPTED with `params_dict()` picking one by hash placement,
        # which is exactly the replace-vs-accumulate hazard the paragraph
        # above says this check exists to close (audit 0.2.0 B6 audit 6).
        # Exact `str` keys make `hash` and `eq` agree by construction.
        names = [k for k, _ in self.params]
        if len(set(names)) != len(names):
            dups = sorted({n for n in names if names.count(n) > 1})
            _load_check(
                False,
                # guarded like every other quote in this pass: `primitive`
                # is document-supplied and a `str` SUBCLASS can refuse
                # `__repr__` (audit 0.2.0 B6 audit 4, F2). A refusal whose
                # own message raises is not a refusal.
                f"JaxprEqn ({_safe_repr(self.primitive)})",
                f"params carry duplicate key(s) {_safe_repr(dups)}: params "
                f"mirrors a jax "
                f"dict, whose keys are unique, and every reader resolves a "
                f"repeat last-wins — so a duplicate silently picks one of two "
                f"contradictory values by serialization order",
            )
        # type-level: a declaration's two self-descriptions must agree
        # (the census's structuralization — see _validate_decl_eqn)
        #
        # AND WHAT THE GUARD VALIDATED IS INSTALLED INTO THE EQUATION —
        # audit 0.2.0 B6 audit 5, F1, and the reason a returned value is
        # not enough. `_validate_decl_eqn` reads the `shape` param ONCE
        # and hands back the extents it compared against the outvar aval;
        # writing them back is what makes every LATER reader — the
        # interval transfer (`propagate`), `_declared_shape`, `_encode`
        # and `coverage.sub_jaxprs` — read the value that was checked,
        # rather than re-reading a self-describing object that is free to
        # answer differently the second time. A `tuple` SUBCLASS whose
        # `__iter__` yielded `(2,)` for the door and `(1,)` for everyone
        # after it passed this check at two elements, was propagated as
        # one, and minted a VERIFIED for `sum(x) <= 3.9` over `x` in
        # `[1,2]^2`, whose true maximum is 4.
        #
        # A SHARED READER CANNOT DO THIS. Routing every reader through one
        # function — the repair `_size`/`_extents` carries one module over
        # — makes every read use one PROTOCOL; it does not make two reads
        # return one VALUE, and `_encode` (which is generic over tuples and
        # cannot know which one is a shape) and `coverage.sub_jaxprs`
        # (which walks params looking for sub-jaxprs and never asks what a
        # param means) could not be routed through it at all.
        #
        # AND THE INSTALL'S OWN COMPARISON IS A READ TOO — audit 0.2.0 B6
        # audit 6, the finding that reopened the one above in the same
        # line. This wrote `(k, dims) if k == "shape" else (k, v)`, a
        # THIRD `==` against a document-supplied key after
        # `_validate_decl_eqn` had made two; a `str` subclass answering
        # True, True, FALSE, then True forever let the door validate the
        # param and report `dims` while this comprehension matched NOTHING
        # and rewrote `params` with the lying object still in it. The
        # sentence that stood here — *"exactly one `shape` key: the
        # duplicate-key refusal above ran"* — was the defect: that refusal
        # guarantees no key appears TWICE, and says nothing about whether
        # this comparison matches ONE. Here it matched zero.
        #
        # It is sound now for two independent reasons, and the second is
        # the one that does not depend on reading the code above: the keys
        # are exact `str` by `_canonical_param_keys`, so `==` is
        # `str.__eq__`; and the count is CHECKED rather than asserted, so
        # a comparison that stops matching what `_validate_decl_eqn`
        # matched is a refusal and not a silent rewrite.
        installed = _validate_decl_eqn(self, "JaxprEqn")
        if installed:
            rewritten = []
            written = 0
            for k, v in self.params:
                if k in installed:
                    written += 1
                    rewritten.append((k, installed[k]))
                else:
                    rewritten.append((k, v))
            if written != len(installed):  # composed only when it fires
                _load_check(
                    False,
                    f"JaxprEqn ({_safe_repr(self.primitive)})",
                    f"the declaration door validated {len(installed)} "
                    f"param(s) {_safe_repr(sorted(installed))} and this "
                    f"equation carries {written} key(s) matching them: the "
                    f"guard's value could not be installed, so a later "
                    f"reader would re-read the object the guard was handed "
                    f"instead of the value it checked",
                )
            object.__setattr__(self, "params", tuple(rewritten))
        # AND EVERY PARAM THE DOOR DOES NOT OWN, canonicalized last: the
        # two it does own are exact already (`_validate_decl_eqn` returns
        # normalised extents and a canonical dtype), so this pass is a
        # no-op on them and is the only door the rest have.
        object.__setattr__(
            self, "params", _canonical_param_values(self.params, "JaxprEqn")
        )

    def params_dict(self) -> dict[str, object]:
        return dict(self.params)


@dataclass(frozen=True)
class DebugInfo:
    func: str = ""
    arg_names: tuple[str, ...] = ()
    result_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # excluded from `content_hash` but not from `to_dict`, so these
        # strings are still serialized and still document-supplied
        # (audit 0.2.0 B6 audit 6)
        _canonicalise(self, "DebugInfo")


@dataclass(frozen=True)
class Jaxpr:
    constvars: tuple[Var, ...]
    invars: tuple[Var, ...]
    outvars: tuple[Atom, ...]  # outputs may be literals, not only vars
    eqns: tuple[JaxprEqn, ...]
    effects: tuple[str, ...] = ()
    debug_info: DebugInfo | None = None

    def __post_init__(self) -> None:
        # canonicalized before the sort, for the reason `JaxprEqn` gives
        # (audit 0.2.0 B6 audit 6)
        _canonicalise(self, "Jaxpr")
        object.__setattr__(self, "effects", tuple(sorted(self.effects)))


@dataclass(frozen=True)
class ClosedJaxpr:
    jaxpr: Jaxpr
    consts: tuple[bool | int | float | complex | str | Array, ...] = field(default=())

    def __post_init__(self) -> None:
        # canonicalized before the pairing below reads either side
        # (audit 0.2.0 B6 audit 6)
        _canonicalise(self, "ClosedJaxpr")
        # local pairing invariant: each const against its constvar's aval
        # (component objects self-validated at their own construction)
        for i, (var, val) in enumerate(zip(self.jaxpr.constvars, self.consts)):
            _validate_value_against_aval(
                val,
                var.aval,
                f"ClosedJaxpr.consts[{i}] (constvar {_safe_repr(var.id)})",
            )

    def to_dict(self, *, include_metadata: bool = True) -> dict:
        return _encode(self, include_metadata)

    @staticmethod
    def from_dict(d: dict) -> "ClosedJaxpr":
        obj = _decode(d)
        if not isinstance(obj, ClosedJaxpr):
            raise ValueError(
                f"expected a serialized ClosedJaxpr, got {_safe_type_name(obj)}"
            )
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
    raise TypeError(f"stelling.ir cannot encode {_safe_type_name(obj)}")


def _decode(obj: object) -> object:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if not isinstance(obj, dict):
        raise ValueError(
            f"malformed IR serialization: unexpected {_safe_type_name(obj)}"
        )
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
    raise ValueError(f"malformed IR serialization: unknown tag {_safe_repr(k)}")


# --- the bounded from_dict validation pass ----------------------------------
#
# See the module docstring: a fixed list of checks at the deserialization
# door, loud (TranscriptionError), symmetric with trace-side
# transcription; NOT a shape-inference engine.
#
# VALIDATION DOES REWRITE, AND REWRITING IS THE POINT — audit 0.2.0 B6
# audit 6, F4. This said *"validation reads the decoded objects and never
# rewrites them"*, and by the time it was read the whole of the preceding
# commit was rewriting them: `Aval`, `Array` and a declaration's `shape`
# param each store the extents their guard validated, and the
# canonicalization door below replaces every stored value with an exact
# twin. What is true is the narrower claim the sentence was reaching for:
# the ENCODE/DECODE FORMS are untouched, and every rewrite is one of the
# canonicalizations named in `CANONICALIZATIONS` — a guard that checks one
# value and stores a different one is precisely the defect this pass
# exists to close, so a guard that could not store what it checked would
# be the wrong shape of guard.

import operator as _operator  # noqa: E402  (stdlib; kept local to the pass)


# THE SHAPE-PARAM CONTAINER RULE, WRITTEN ONCE — audit 0.2.0 B6 audit 4, F1.
#
# A declaration records its extents in one of these types. This tuple is the
# WHOLE of the rule: :func:`_validate_decl_eqn` below refuses a ``shape``
# param that is not one of them, and
# :meth:`stelling.obligation._Slicer._declared_shape` declines it — both by
# asking ``isinstance`` against THIS object, and neither by restating the
# list. Two copies of a rule are two rules, and the whole of this batch is
# about a claim that stopped matching the code beside it.
#
# WHY A POSITIVE TEST rather than a list of refused containers. The refusal
# used to enumerate the types the hazard was first noticed in —
# ``(str, bytes, bytearray)`` — and ``memoryview`` and ``array.array``
# walked past it carrying the identical misread: ``tuple(memoryview(b"34"))``
# is ``(51, 52)``, a pair of perfectly plausible extents the declaration
# never said, and the door ACCEPTED that form. Adding two more names is
# "the container type I happened to enumerate", which
# :func:`_validate_param_value` is annotated in this same batch as
# condemning. Naming the two forms a declaration is actually recorded in
# makes every other one fall out of the rule instead of having to be named
# by it — including the one nobody has thought of yet.
#
# WHY EXACTLY THESE TWO, and what the narrowing costs. They are the only
# forms that reach here from a document anyone can produce without hand-
# building IR: :func:`_decode` builds a ``tuple`` for every sequence param,
# jax's own params carry tuples and lists, and ``harness.any_array``
# normalises whatever the caller passes into a ``tuple`` before the
# primitive is bound (measured: twelve exotic ``shape`` arguments — a
# ``range``, an ``array.array``, a numpy array, a custom iterable — all
# trace to a ``tuple`` param). So the narrowing is confined to HAND-BUILT
# IR, and there it is loud: a `TranscriptionError` naming the type.
#
# `tests/test_shape_param_rule.py` measures the accept/refuse partition of
# BOTH faces against this tuple, over a population of container types it
# COMPUTES rather than lists, so the rule and the behaviour cannot drift
# apart in silence. That test is the mechanism; this comment is not.
_SHAPE_PARAM_CONTAINERS: tuple[type, ...] = (tuple, list)

# The rule as a sentence, DERIVED from the tuple above so a refusal can
# never name a different set from the one it applied.
_SHAPE_PARAM_RULE = " or ".join(f"a {t.__name__}" for t in _SHAPE_PARAM_CONTAINERS)


def _safe_repr(obj) -> str:
    """``repr(obj)``, or a visible placeholder when the object refuses.

    For quoting an object this pass is describing, and for nothing else.
    `__post_init__` is the public constructor's own body, so a `repr` that
    raises inside a message composed here leaves `ir.JaxprEqn(...)` as a
    raw `RuntimeError` — which is precisely the class `_validate_decl_eqn`
    exists to close, arriving through the sentence that was meant to close
    it (audit 0.2.0 B6 audit 3, F3). Measured: a well-formed declaration
    whose extent has a working `__index__` and a refusing `__repr__`
    raw-crashed the constructor, because a `_load_check` message is an
    ARGUMENT and is therefore composed whether or not the check fails.

    EVERY object-valued quote in this pass goes through this function or
    one of its two siblings (:func:`_safe_type_name`, :func:`_safe_str`),
    including the ones in a `where` string and the ones on a branch that
    only composes when the check has already failed — audit 0.2.0 B6
    audit 4, F2. The record's figure is **10 distinct quote sites** — ten
    INTERPOLATIONS of an object into a message — of which the previous
    audit had named four; two of the six it had not fire on the PASSING
    path (`ir.Array(dtype=<a str subclass whose repr raises>)` and
    `ir.Literal(val=<an int subclass whose repr raises>)`, both
    well-formed documents).

    THE SWEEP'S OWN UNIT IS THE MESSAGE EXPRESSION, and this paragraph
    recorded two units as one — audit 0.2.0 B6 audit 5, F2. A
    `_load_check` message spans several lines and can interpolate on more
    than one, so a per-LINE count is larger than a per-MESSAGE count, and
    a per-QUOTE count (the ten above) is larger again. The three
    measurements, each computed by `tests/test_ir_message_totality.py`
    rather than trusted from this paragraph:

        this tree, as shipped              95 swept / 0 escapes / 20 skipped
        guards neutered                     1 escape  /  1 line  / 1 message
        guards neutered, and the door's
          LEAF READS neutered too          26 escapes /  8 lines / 8 messages
        `30d4b04`, guards absent           28 escapes / 10 lines / 8 messages

    The sweep's 8 message expressions plus the 2 the canonical document
    masks also comes to 10; that is a different quantity from the ten
    quotes above and agrees with it by arithmetic, not by derivation. A
    guarded quote is cheap; deciding per site which objects can misbehave
    is how those six were missed.

    **THE GUARD-NEUTERED FIGURES ARE MEASURED WITH THE CANONICALIZATION
    DOOR REMOVED TOO** — audit 0.2.0 B6 audit 6. Every hostile leaf that
    sweep injects is a SUBCLASS of a stored type, and the door now
    replaces one with an exact twin before any message quotes it: 25 of
    the 26 ESCAPES therefore do not happen, across 7 of the 8 message
    expressions, and they are not guarded — they are unreachable. (Two
    units, both stated, for the reason the rest of this docstring gives.)
    A control that
    neutered only the guards measured 1 escape and would have gone on
    passing with every call to this function deleted, so it neuters the
    door's leaf reads as well and reports each measurement with the other
    out of the way. This function is still
    load-bearing at the one site canonicalization does not own — the
    hostile extent inside a declaration's `shape` param, which
    `_validate_decl_eqn` reads as handed in because it has its own rule
    for that param."""
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 — the message's own totality
        return "<unreadable>"


def _safe_type_name(obj) -> str:
    """``type(obj).__name__``, or a visible placeholder — the sibling of
    :func:`_safe_repr` for the refusals that name the type they refused."""
    try:
        return type(obj).__name__
    except Exception:  # noqa: BLE001 — the message's own totality
        return "<unreadable>"


def _safe_str(obj) -> str:
    """``str(obj)``, or a visible placeholder — the sibling of
    :func:`_safe_repr` for a `where` PATH SEGMENT, which reads as a name
    rather than as a quotation. Found by the sweep in
    `tests/test_ir_message_totality.py`, not by reading: a
    `NamedTupleParam` field name is document-supplied, and interpolating
    it bare took `str.__format__` — a third read again — straight out of
    `ClosedJaxpr.from_dict`."""
    try:
        return str(obj)
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


# --- the canonicalization door ----------------------------------------------
#
# WHAT A DOCUMENT SUPPLIES, THIS MODULE STORES AS AN EXACT BUILT-IN — or
# refuses it. Audit 0.2.0 B6 audit 6.
#
# THE CLASS, AND WHY IT WAS NOT CLOSED BY FIXING ITS MEMBERS. Five repairs
# before this one each routed one PAIR of reads onto one value: the
# ``shape`` param validated by ``tuple()`` and consumed by ``tuple()``
# again (audit 5, F1); the param KEY compared by ``==`` twice in
# :func:`_validate_decl_eqn` and a third time by the install that read it
# back (audit 6); the same key counted by ``hash`` through ``set()`` and
# by ``eq`` through ``list.count`` in the duplicate refusal, so that two
# keys with equal text and different ``__hash__`` were not a duplicate;
# the ``dtype`` param compared with ``==`` here and consumed with
# ``str()`` in ``propagate._ieee_any``. A fifth family — ``axes``,
# ``new_sizes``, ``slice_sizes``, ``dimension_numbers`` — had no door at
# all. Each repair was correct and none of them closed anything: the
# members are found by whoever last looked, and three consecutive rounds
# of this batch ended with a new one.
#
# The fact underneath every member is that Python's protocols are
# OVERRIDABLE, so a document-supplied object may answer ``==``, ``hash``,
# ``iter``, ``str``, ``index``, ``len`` or ``getitem`` differently on two
# reads, and a guard that checked one answer has not checked the other.
# The repair that does not need the members enumerated is to leave nothing
# overridable in what is stored: every value is replaced, at construction,
# by an EXACT instance of a type this module is closed over. A later read
# then cannot differ, because there is no subclass left to answer it — and
# that is a property of the STORED OBJECT rather than of any reader, so it
# holds for readers nobody has written yet, which is what the per-member
# repairs could not do.
#
# WHERE THE BOUNDARY IS BETWEEN A VALUE THAT DECIDES AND ONE THAT IS
# MERELY CARRIED: there is not one, and looking for it is how this class
# stayed open. ``dtype`` was a carried param until ``propagate._ieee_any``
# began selecting a subnormal band from it; ``update_jaxpr``'s mere
# PRESENCE became semantic when the scatter-add row learned to read it.
# The rule here is therefore uniform over every field of every dataclass
# in this module — the fields are read from :func:`dataclasses.fields`,
# not listed, so a field added later is covered without anyone having to
# remember this comment.
#
# A SUBCLASS IS READ, NOT REFUSED, AND THE TRACE PATH IS WHY. Refusing
# everything but an exact type would be simpler and would refuse traced
# queries: ``np.float64`` IS a ``float`` subclass and
# ``_jax_compat.Transcriber.param`` returns it unchanged from its
# ``isinstance(v, (int, float, complex))`` arm, and ``np.str_`` IS a
# ``str`` subclass leaving the ``isinstance(v, (bool, str))`` arm the same
# way. Both are measured. So a subclass is CANONICALIZED, by one read, and
# only a type with no exact form to store is refused.
#
# THE READ IS THE BASE TYPE'S OWN ACCESSOR, NOT THE PUBLIC PROTOCOL.
# ``str.__str__``, ``int.__index__``, ``float.__float__``,
# ``bytes.__getitem__`` and ``tuple.__getitem__`` reach the payload the
# instance actually carries, return an exact instance of the base type,
# and cannot be redirected by an override. ``str(v)``, ``int(v)`` and
# ``tuple(v)`` are each one read too and would be sound for the same
# reason, but they ask the object how it would like to be read; this
# module transcribes what an object IS. An EXACT value never reaches one
# of these readers at all — the dispatch below is on ``type(v)`` and
# returns it unchanged — so the traced and deserialized routes, whose
# values are exact already, pay one set lookup per value and allocate
# nothing.

import dataclasses as _dataclasses  # noqa: E402  (stdlib; kept local to the pass)


class _NotCanonical(Exception):
    """Internal signal: an object this module has no exact form to store.

    Carries the object and the path back up to the field it was found
    under, appended innermost-first as the signal propagates, so the
    refusal :func:`_refuse_uncanonical` composes names both.
    """

    def __init__(self, obj: object) -> None:
        super().__init__()
        self.obj = obj
        self.path: list[str] = []


# THE READS, ONE PER BASE TYPE, IN THIS ORDER. ``bool`` is an ``int``
# subclass and must claim the `isinstance` before ``int`` does, or
# ``int.__index__`` would store ``True`` as ``1`` and change what a param
# says; ``bool`` cannot itself be subclassed, so its read is the identity
# and its entry exists only to take that place in the order.
_CANONICAL_READS: tuple[tuple[type, object], ...] = (
    (bool, lambda v: v),
    (int, int.__index__),
    (float, float.__float__),
    (complex, lambda v: complex(complex.real.__get__(v),
                                complex.imag.__get__(v))),
    (str, str.__str__),
    (bytes, lambda v: bytes.__getitem__(v, slice(None))),
)

# The leaf types that are already what they will be stored as when their
# type is EXACT. Derived from the reads above so the two cannot name
# different sets.
_CANONICAL_EXACT: frozenset[type] = frozenset(
    (type(None),) + tuple(t for t, _ in _CANONICAL_READS)
)

# The rule as a sentence, DERIVED from the table, so a refusal can never
# name a different set from the one it applied — the same discipline
# :data:`_SHAPE_PARAM_RULE` carries. ``_CANONICAL_IR_TYPES`` is computed
# at the foot of this module from the module's own dataclasses.
_CANONICAL_RULE = (
    ", ".join(["None"] + [t.__name__ for t, _ in _CANONICAL_READS] + ["tuple"])
    + " (a list is stored as a tuple), or one of stelling.ir's own frozen "
    "dataclasses"
)


# TYPES ANOTHER `stelling` MODULE BUILDS AND THIS ONE CARRIES. Exactly
# one today: `interval.IntervalArray`, which `ClosedJaxpr.consts` may hold
# in place of a value — "a const already handed over as a box, provenance
# unknown", the form `propagate._Propagator.run` binds directly and
# `SOUNDNESS.md` records with its own test.
#
# WHY IT IS A SEAM AND NOT A NAME. `ir` may not import anything outside
# the standard library, so it cannot name that class; and it must not
# guess by duck-typing, because "an object that looks like a box" is the
# enumeration this whole door replaces. Registration is safe against
# import order for a reason particular to it: a caller holding an
# `IntervalArray` has necessarily imported `stelling.interval`, so the
# registration has necessarily run.
#
# WHAT REGISTRATION DELEGATES, stated so it is not mistaken for a hole: a
# registered type is CARRIED, not canonicalized — the registering module
# owns its single-valuedness (`IntervalArray` is a frozen dataclass whose
# own `__post_init__` validates it). No DOCUMENT can reach this arm:
# `_decode` has no tag for such a type and `_encode` refuses to encode
# one, so a registered value can only come from a caller who built it,
# and it is outside `content_hash` and `to_dict` entirely.
_LIBRARY_STORED_TYPES: tuple[type, ...] = ()


def _register_stored_type(cls: type) -> type:
    """Declare ``cls`` a library-supplied type :func:`_canonical` carries.

    Package-internal (see the comment above for what it delegates).
    Returns ``cls`` so it can be used as a decorator.
    """
    global _LIBRARY_STORED_TYPES
    if cls not in _LIBRARY_STORED_TYPES:
        _LIBRARY_STORED_TYPES = _LIBRARY_STORED_TYPES + (cls,)
    return cls


def _canonical(obj: object) -> object:
    """``obj`` as an EXACT instance of a type this module stores.

    Raises :class:`_NotCanonical` — never a `TranscriptionError` — because
    the path back to the field is composed as the signal unwinds and only
    the caller that owns a `where` can turn it into a refusal.
    """
    t = type(obj)
    if t in _CANONICAL_EXACT:
        return obj
    if t is tuple:
        return _canonical_items(obj)
    if t is list:
        # A LIST IS STORED AS A TUPLE. An exact `list` is an exact
        # built-in and is still two answers waiting to happen, because it
        # is MUTABLE: nothing stops the document's own reference to it
        # being appended to after `__post_init__` returned. `_encode` also
        # has no `list` arm, so a `list` that survived this door produced
        # IR that `content_hash()` and `to_dict()` could not serialize at
        # all; the trace path already collapses list to tuple
        # (`_jax_compat.Transcriber.param`) and `_decode` never builds
        # one, so the conversion costs no route anything.
        return _canonical_items(tuple(obj))
    if t in _CANONICAL_IR_TYPES:
        # canonical by its OWN `__post_init__`, which ran before this
        # object could be handed to anything — so there is nothing to
        # recurse into, and a document costs one pass over its own size
        # rather than one per level of nesting
        return obj
    if t in _LIBRARY_STORED_TYPES:
        # carried, by the declaration another `stelling` module made — and
        # checked HERE, before the subclass reads below, because
        # registration says "carry this" and a registered type that
        # happened to subclass a stored one would otherwise be rewritten
        # into something its own module did not build
        return obj
    for base, read in _CANONICAL_READS:
        if isinstance(obj, base):
            return read(obj)
    if isinstance(obj, tuple):
        return _canonical_items(tuple.__getitem__(obj, slice(None)))
    if isinstance(obj, list):
        return _canonical_items(tuple(list.__getitem__(obj, slice(None))))
    raise _NotCanonical(obj)


def _canonical_items(items: tuple) -> tuple:
    """Every element of an EXACT tuple, canonicalized; ``items`` itself
    when nothing moved, so an already-canonical document is not copied."""
    out: list = []
    changed = False
    for i, x in enumerate(items):
        try:
            c = _canonical(x)
        except _NotCanonical as exc:
            exc.path.append(f"[{i}]")
            raise
        changed = changed or c is not x
        out.append(c)
    return items if not changed else tuple(out)


def _canonical_shell(v: object) -> tuple:
    """An EXACT tuple of ``v``'s items, with the items left alone.

    For the two places that need the CONTAINER canonical before its
    contents are: the ``params`` sequence and each ``(key, value)`` entry
    in it, whose keys must be settled before the sort and whose values
    must not be, because :func:`_validate_decl_eqn` owns two of them.
    """
    t = type(v)
    if t is tuple:
        return v
    if t is list:
        return tuple(v)
    if isinstance(v, tuple):
        return tuple.__getitem__(v, slice(None))
    if isinstance(v, list):
        return tuple(list.__getitem__(v, slice(None)))
    raise _NotCanonical(v)


def _refuse_uncanonical(exc: _NotCanonical, where: str) -> None:
    """Turn a :class:`_NotCanonical` into the module's `TranscriptionError`.

    Both quotes guarded: the object being refused is document-supplied by
    definition, and a `str` subclass whose `__repr__` raises is exactly
    the kind of object that reaches here (audit 0.2.0 B6 audit 4, F2).
    """
    _load_check(
        False,
        f"{where}{''.join(reversed(exc.path))}",
        f"value {_safe_repr(exc.obj)} is of type "
        f"{_safe_type_name(exc.obj)}, which this module has no exact form "
        f"to store: a document-supplied value is kept as an EXACT "
        f"{_CANONICAL_RULE}, so that no reader after a guard can be handed "
        f"a different answer than the guard checked — and a type with no "
        f"exact form to store has none to check either",
    )


_CANONICAL_FIELDS: dict[type, tuple[str, ...]] = {}


def _canonicalise(obj, where: str, *, skip: tuple[str, ...] = ()) -> None:
    """Replace every field of an IR dataclass with its exact-typed twin.

    The field names come from :func:`dataclasses.fields`, cached per
    class: a field added to one of these dataclasses later is canonicalized
    without anyone having to add it here, which is the difference between
    this and a repair per member.
    """
    cls = type(obj)
    names = _CANONICAL_FIELDS.get(cls)
    if names is None:
        names = _CANONICAL_FIELDS[cls] = tuple(
            f.name for f in _dataclasses.fields(cls)
        )
    for name in names:
        if name in skip:
            continue
        v = getattr(obj, name)
        try:
            c = _canonical(v)
        except _NotCanonical as exc:
            exc.path.append(f".{name}")
            _refuse_uncanonical(exc, where)
        if c is not v:
            object.__setattr__(obj, name, c)


def _canonical_param_keys(params, where: str) -> tuple[tuple[str, object], ...]:
    """The ``params`` sequence, its entries and its KEYS — canonicalized;
    the values left for :func:`_canonical_param_values` to take after
    :func:`_validate_decl_eqn` has had the two it owns.

    THE KEYS MOVE FIRST BECAUSE THE SORT READS THEM. `JaxprEqn` stores
    params sorted (a commitment — see the module docstring), and a `str`
    SUBCLASS that overrides the ordering protocol decides that order
    itself: measured on `f729d70`, three honest params `aaa`/`bbb`/`ccc`
    whose first key returns True from `__gt__` stored as `bbb, ccc, aaa`.
    Everything downstream of the sort reads a key too — the duplicate
    refusal's `set()`/`count()`, `dict()`, `_validate_decl_eqn`'s
    ``"shape" in params``, and the install's own comparison.

    A KEY MUST BE AN EXACT `str` AFTER THAT READ, not merely canonical:
    the field is declared ``tuple[tuple[str, object], ...]``, every
    consumer indexes :meth:`JaxprEqn.params_dict` by name, and a mixed
    ``int``/``str`` key set makes the sort itself raise
    (``'<' not supported between instances of 'str' and 'int'``). Neither
    non-hand-built route can produce one: `_decode` reads each key
    straight out of a JSON string and the trace path reads a jax
    ``dict``'s own keys.
    """
    try:
        entries = _canonical_shell(params)
    except _NotCanonical as exc:
        exc.path.append(".params")
        _refuse_uncanonical(exc, where)
    out: list[tuple[str, object]] = []
    for i, entry in enumerate(entries):
        try:
            pair = _canonical_shell(entry)
        except _NotCanonical:
            pair = None
        # ONE REFUSAL FOR BOTH SHAPES OF THE SAME FACT — an entry that is
        # not a `tuple`/`list` and an entry of the wrong length are both
        # "not a (key, value) pair", and neither is refused for being an
        # uncanonicalizable VALUE, which is a different sentence about a
        # different object.
        #
        # COMPOSED ONLY ON THE FAILING PATH, and deliberately so: a
        # `_load_check` message is an ARGUMENT, and this one quotes a param
        # entry — which may hold a whole nested `ClosedJaxpr` — so composing
        # it on every construction of every equation is a cost the guarded-
        # quote rule does not speak to. This is the one place in the pass
        # where the quoted object has no size bound.
        if pair is None or len(pair) != 2:
            _load_check(
                False,
                f"{where}.params[{i}]",
                f"params entry {_safe_repr(entry)} is not a (key, value) "
                f"pair: params mirrors a jax dict's ITEMS, and everything "
                f"here unpacks one into a key and a value — `dict()` and "
                f"`for k, v in ...` will both take a 2-character `str` and "
                f"read a key and a value the document never stated, so "
                f"'it unpacks' is not the test",
            )
        try:
            key = _canonical(pair[0])
        except _NotCanonical as exc:
            exc.path.append(f".params[{i}] key")
            _refuse_uncanonical(exc, where)
        if type(key) is not str:
            _load_check(
                False,
                f"{where}.params[{i}]",
                f"params key {_safe_repr(pair[0])} is of type "
                f"{_safe_type_name(pair[0])}, and a param key is a NAME: "
                f"`params` is declared `tuple[tuple[str, object], ...]`, the "
                f"stored order is a sort over these keys, and every consumer "
                f"indexes `params_dict()` by name",
            )
        out.append((key, pair[1]))
    return tuple(out)


def _canonical_param_values(params, where: str) -> tuple[tuple[str, object], ...]:
    """The param VALUES, canonicalized — every param that has no rule of
    its own, which is every one but the two :func:`_validate_decl_eqn`
    owns, and those two as well, where this is a no-op because what that
    function installed is exact already.

    THIS IS WHAT CLOSES THE PARAMS NO DOOR NAMES. ``axes``,
    ``new_sizes``, ``slice_sizes`` and ``dimension_numbers`` are read by
    the transfers and by the emission with no validation between them and
    the document, and this function does not validate them either: giving
    them a schema would be per-primitive shape inference, which `ir.py`
    scopes out in writing. What it does is narrower and is the whole of
    this class — it makes each of them SINGLE-VALUED, so the transfer and
    the emission read the same extents whatever protocol either asks. That
    they are the RIGHT extents is a different claim, and is still not made
    here.
    """
    out: list[tuple[str, object]] = []
    changed = False
    for k, v in params:
        try:
            c = _canonical(v)
        except _NotCanonical as exc:
            exc.path.append(f".params[{_safe_repr(k)}]")
            _refuse_uncanonical(exc, where)
        changed = changed or c is not v
        out.append((k, c))
    return params if not changed else tuple(out)


def _validate_array_value(arr: Array, where: str) -> tuple[int, ...]:
    """Validate an :class:`Array`'s self-description, and RETURN the
    normalised extents it validated — audit 0.2.0 B6 audit 3, F1, the
    same shape of repair as :func:`_load_extents`. The byte-length product
    below and :func:`_validate_value_against_aval`'s comparison above both
    take what this read, never a second `int(d)` read of the same
    self-describing objects."""
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
            # `arr.dtype` is a document-supplied object, and a `str`
            # SUBCLASS whose `__repr__` raises satisfies every check above
            # it: this message is an ARGUMENT, so an unguarded `{!r}` here
            # raw-crashed `ir.Array(...)` on a WELL-FORMED array (audit
            # 0.2.0 B6 audit 4, F2 — a site the previous pass did not name)
            f"array value of shape {extents} dtype {_safe_repr(arr.dtype)} "
            f"carries {len(arr.data)} byte(s), expected {n * itemsize}",
        )
    return extents


def _validate_value_against_aval(val, aval: Aval, where: str) -> None:
    # BOTH SIDES THROUGH THE SAME READER, so the comparison is int-to-int
    # and every extent quoted below is a plain `int` — audit 0.2.0 B6
    # audit 4, F2. This compared `tuple(val.shape) == tuple(aval.shape)`,
    # which asks the raw objects' own `__eq__` after nothing validated
    # them here, and quoted them unguarded into a message that is an
    # ARGUMENT to `_load_check`: an extent with a working `__index__` and
    # a refusing `__repr__` therefore raw-crashed `ir.Literal(...)` and
    # `ir.ClosedJaxpr(...)` on the PASSING path. The aval's own
    # `__post_init__` has validated its extents already; this read is
    # defensive rather than covered, because this function may not rest on
    # another guard having run — the same reason `_validate_decl_eqn`
    # re-reads.
    aval_problem, aval_dims = _load_extents(aval.shape)
    _load_check(aval_problem is None, where, f"aval has {aval_problem}")
    if isinstance(val, Array):
        val_dims = _validate_array_value(val, where)
        _load_check(
            val_dims == aval_dims,
            where,
            f"array value shape {val_dims} contradicts the "
            f"recorded aval shape {aval_dims}",
        )
    elif isinstance(val, (bool, int, float, complex)):
        _load_check(
            aval_dims == (),
            where,
            # a document-supplied scalar, and `isinstance` is a claim
            # about the TYPE: an `int` subclass whose `__repr__` raises
            # satisfies it (audit 0.2.0 B6 audit 4, F2)
            f"scalar value {_safe_repr(val)} under a non-scalar aval "
            f"shape {aval_dims}",
        )
    # str and None values carry no shape claim to cross-check


def _validate_aval(aval: Aval, where: str) -> tuple[int, ...]:
    """Validate an :class:`Aval`'s extents and RETURN them normalised.

    :meth:`Aval.__post_init__` installs what this returns, so the shape an
    aval carries after construction is the shape this function passed —
    audit 0.2.0 B6 audit 5, F1. The predicate spelling
    (:func:`_load_extent_problem`) tested each extent and discarded it,
    leaving every later reader of ``aval.shape`` — `_encode` here,
    `obligation._shape_of` and `_size`, `propagate._atom_element_count`
    and `_declared_element_count` — to re-read the raw objects."""
    problem, extents = _load_extents(aval.shape)
    _load_check(problem is None, where, f"aval has {problem}")
    return extents


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
            _validate_param_value(item, f"{where}.{_safe_str(name)}")


def _validate_jaxpr(jaxpr: Jaxpr, where: str) -> None:
    for i, v in enumerate(jaxpr.constvars):
        _validate_aval(v.aval, f"{where}.constvars[{i}]")
    for i, v in enumerate(jaxpr.invars):
        _validate_aval(v.aval, f"{where}.invars[{i}]")
    for i, a in enumerate(jaxpr.outvars):
        _validate_atom(a, f"{where}.outvars[{i}]")
    for k, eqn in enumerate(jaxpr.eqns):
        # a `where` string is a message too: every object-valued quote in
        # this pass goes through `_safe_repr` (audit 0.2.0 B6 audit 4, F2)
        w = f"{where}.eqns[{k}] ({_safe_repr(eqn.primitive)})"
        for i, a in enumerate(eqn.invars):
            _validate_atom(a, f"{w}.invars[{i}]")
        for i, v in enumerate(eqn.outvars):
            _validate_aval(v.aval, f"{w}.outvars[{i}]")
        params = dict(eqn.params)
        for name, v in eqn.params:
            _validate_param_value(v, f"{w}.params[{_safe_repr(name)}]")
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
        f"missing param(s) {missing} that jax supplies on every "
        f"{_safe_repr(eqn.primitive)} "
        f"equation; absence is not a traced form, and readers that test key "
        f"PRESENCE would take it for a hand-built equation and apply the "
        f"primitive's default meaning",
    )


def _validate_decl_eqn(eqn: "JaxprEqn", where: str) -> dict[str, object]:
    """A stelling_any declaration's aval must agree with its OWN
    params — two self-descriptions of one declared set. Called from
    JaxprEqn.__post_init__ (every construction path) and from the
    from_dict walk (kept for its load-path error context).

    **RETURNS THE PARAMS IT VALIDATED, NORMALISED, KEYED BY NAME** — the
    ``shape`` param's extents as plain ``int`` in a plain ``tuple`` and
    the ``dtype`` param as an exact ``str`` — and an EMPTY dict when the
    equation carries neither to validate (a form this function
    deliberately still blesses; see the last paragraph).
    :meth:`JaxprEqn.__post_init__` writes each back into ``params``, which
    is what makes the value every later reader sees the value this
    function compared. Audit 0.2.0 B6 audit 5, F1: reading once and
    binding LOCALLY was audit 3's F1 repair and it is correct as far as it
    goes, but the binding died with the call, and the transfer in
    `propagate`, `_encode`, `coverage.sub_jaxprs` and
    :meth:`stelling.obligation._Slicer._declared_shape` then each re-read
    the raw param. Two reads of a self-describing object are two answers
    it is free to make different.

    **THE ``dtype`` PARAM IS THE SAME FINDING ONE PARAM OVER** — audit
    0.2.0 B6 audit 6. This function compared ``dtype == aval.dtype``,
    which a ``str`` SUBCLASS satisfies (it satisfies the ``isinstance``
    above the comparison too), while ``propagate._ieee_any`` consumes
    ``str(_req(params, "dtype", ...))`` and selects the subnormal band
    from THAT — so overriding ``__str__`` alone showed the door one dtype
    and the ieee declaration transfer another. Measured on this IR: a
    param whose ``repr`` is ``'float64'`` and whose ``str()`` is
    ``'int64'`` was accepted, and the transfer took the arm for a
    declaration with no float format at all. It is canonicalized and
    installed here for the same reason the extents are.

    **THE RULE ON THE ``shape`` PARAM IS POSITIVE, AND IT IS THE PARAM'S
    CONTAINER TYPE:** a declaration records its extents in one of
    :data:`_SHAPE_PARAM_CONTAINERS`, and a ``shape`` param of any other
    type is REFUSED — not skipped, and not coerced. The tuple is the whole
    of the rule and this sentence restates none of it; the two branches
    below ask ``isinstance`` against that object, and
    :meth:`stelling.obligation._Slicer._declared_shape` asks the same
    object rather than keeping a second copy of the answer.

    That is a NARROWING, and the narrowing is the point. This check used
    to run only ``if isinstance(shape, tuple)``, so a ``list`` param
    skipped it entirely and a declaration saying four elements in its
    param and two in its aval was constructible (audit 0.2.0 B6 re-audit,
    UNSOUND-1); the enumeration that replaced it — refuse ``str``,
    ``bytes``, ``bytearray``, read anything else — let ``memoryview`` and
    ``array.array`` through with the misread the enumeration existed to
    stop. What is refused now is everything the rule does not name, so
    ``range``, ``array.array``, ``memoryview``, a numpy array and a custom
    iterable are all refused, and so is whichever sequence type is noticed
    next. **Being a sequence of extents is not the test and never was the
    test after this commit** — audit 0.2.0 B6 audit 4, F1, where this
    paragraph still described the enumeration two commits after the code
    stopped implementing it.

    What that costs is stated where the rule is (see
    :data:`_SHAPE_PARAM_CONTAINERS`): every route that is not hand-built
    IR normalises to a ``tuple`` before this is reached, so the refusal is
    confined to hand-built IR, and there it names the type it refused.

    Two distinct failures, and they are refused SEPARATELY because they
    are different facts about the document — the same split
    :meth:`_declared_shape` makes. A param of the wrong container type is
    refused for its TYPE; a param of the right type whose iteration raises
    (a ``list`` SUBCLASS whose ``__iter__`` refuses — ``isinstance`` is a
    claim about the type and not about the object) is refused for being
    UNREADABLE. One sentence for both said "is not a sequence of extents"
    over ``np.array([4])``, which is a sequence of extents and was refused
    for a different reason.

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
        return {}
    params = dict(eqn.params)
    normalised: dict[str, object] = {}
    # the declaration's aval must agree with its OWN params —
    # the P1 arc's lies all started at a declaration whose two
    # self-descriptions disagreed
    if "shape" in params:
        shape = params["shape"]
        # THE RULE, ASKED OF THE ONE OBJECT THAT HOLDS IT. The slicer's
        # `_declared_shape` asks `ir._SHAPE_PARAM_CONTAINERS` too, so the
        # two faces cannot part company by one of them being edited.
        held_in_a_container = isinstance(shape, _SHAPE_PARAM_CONTAINERS)
        dims: tuple | None = None
        if held_in_a_container:
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
        # THE QUOTES ON THIS BRANCH ARE GUARDED, because a `_load_check`
        # message is an ARGUMENT and is therefore composed on the passing
        # path too: an unguarded `{shape!r}` raw-crashed `JaxprEqn(...)` on
        # a document with nothing wrong with it, whose extent merely had a
        # `__repr__` that refuses (audit 0.2.0 B6 audit 3, F3). That
        # sentence was written as *"every quote here is guarded"* and was
        # false 44 lines below itself — the `dtype` arm at the end of this
        # function was not, and neither were three sites in the rest of the
        # pass (audit 0.2.0 B6 audit 4, F2). The claim is narrowed to what
        # it covers, the other sites are fixed, and the sweep for the class
        # is `tests/test_ir_message_totality.py` rather than a sentence.
        #
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
        # THE TWO FAILURES ARE NAMED SEPARATELY (audit 0.2.0 B6 audit 4,
        # F1): the wrong container type is a fact about the param's TYPE,
        # and one sentence for both told a reader holding `np.array([4])`
        # that it "is not a sequence of extents" — which it is, and which
        # is not why it was refused.
        _load_check(
            held_in_a_container,
            where,
            f"stelling_any shape param {_safe_repr(shape)} is of type "
            f"{_safe_type_name(shape)}: a declaration records its extents "
            f"in {_SHAPE_PARAM_RULE}, and reading any other container as "
            f"one would model an array the declaration never described "
            f"(`tuple(b\"34\")` is `(51, 52)`), so it cannot be checked "
            f"against the outvar aval shape {aval_dims} it is the second "
            f"description of",
        )
        _load_check(
            dims is not None,
            where,
            f"stelling_any shape param {_safe_repr(shape)} is "
            f"{_SHAPE_PARAM_RULE} whose iteration RAISES, so it is not a "
            f"readable sequence of extents and cannot be checked against "
            f"the outvar aval shape {aval_dims} it is the second "
            f"description of",
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
        # what the caller installs: the extents this function READ, not the
        # object it read them from (audit 0.2.0 B6 audit 5, F1)
        normalised["shape"] = param_dims
    raw_dtype = params.get("dtype")
    if isinstance(raw_dtype, str) and eqn.outvars[0].aval.dtype is not None:
        # THE COMPARISON IS BETWEEN EXACT STRINGS, and the exact string is
        # what the caller installs — audit 0.2.0 B6 audit 6, the `dtype`
        # sibling of the extents above. `isinstance` is a claim about the
        # TYPE: a `str` subclass satisfies it, satisfies `str.__eq__`
        # against the aval, and still hands `str()` a different answer to
        # `propagate._ieee_any`, which picks the subnormal band from it.
        # `str.__str__` reads the character data the instance carries and
        # cannot be redirected (see the canonicalization door).
        dtype = str.__str__(raw_dtype)
        _load_check(
            dtype == eqn.outvars[0].aval.dtype,
            where,
            # both quotes guarded: `dtype` passes `isinstance(dtype, str)`
            # as a str SUBCLASS whose `__repr__` raises, and the aval's own
            # dtype is document-supplied too — this arm sat 44 lines under
            # a comment claiming every quote in this function was guarded
            # (audit 0.2.0 B6 audit 4, F2)
            f"stelling_any dtype param {_safe_repr(dtype)} contradicts the "
            f"outvar aval dtype {_safe_repr(eqn.outvars[0].aval.dtype)}",
        )
        normalised["dtype"] = dtype
    return normalised


def _validate_closed(closed: ClosedJaxpr, where: str = "query") -> None:
    _validate_jaxpr(closed.jaxpr, where)
    for i, (var, val) in enumerate(zip(closed.jaxpr.constvars, closed.consts)):
        _validate_value_against_aval(
            val, var.aval, f"{where}.consts[{i}] (constvar {_safe_repr(var.id)})"
        )


def _validate_loaded(closed: ClosedJaxpr) -> None:
    """The bounded from_dict validation pass (module docstring)."""
    _validate_closed(closed)


# THE MODULE'S OWN DATACLASSES, COMPUTED — the third arm of
# :func:`_canonical`'s dispatch, and the one that must not be a list. A
# class defined in this module as a FROZEN dataclass has canonicalized its
# own fields in its own `__post_init__` and cannot be reassigned
# afterwards, which is the whole of what that arm rests on; frozen-ness is
# therefore tested rather than assumed, so a mutable dataclass added here
# later does not silently join a set whose members are relied on to be
# single-valued. Written at the foot because it is computed from the
# module namespace; `_canonical` reads it at CALL time, and nothing in
# this module constructs IR at import.
_CANONICAL_IR_TYPES: frozenset[type] = frozenset(
    obj
    for obj in tuple(globals().values())
    if isinstance(obj, type)
    and _dataclasses.is_dataclass(obj)
    and obj.__module__ == __name__
    and obj.__dataclass_params__.frozen
)
