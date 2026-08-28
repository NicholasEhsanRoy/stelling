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
stayed open). Loaded IR is checked for: **the type the code declares at every position
it stores a document value at** — a dataclass field's own annotation,
read with :func:`typing.get_type_hints` rather than listed, and the
container :func:`_encode` writes at every sequence position the reader
has to iterate (see the document-schema narrative below ``_encode``);
shape entries integral and
nonnegative everywhere (avals, ``stelling_any`` shape params,
:class:`Array` value shapes); ``stelling_any`` outvar avals consistent
with their own shape/dtype params, and its ``lo``/``hi`` a binary64 pair
declaring a NON-EMPTY set (the refusal ``harness.any_array`` makes at the
trace face, made here too — this one on the LOAD path only, because a
document claims to be a query that came through that face while
``ir.JaxprEqn`` is the constructor underneath it);
:class:`Array` payload lengths equal
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
    (
        "complex parts",
        "A `complex` value's `re`/`im` are recorded as binary64 and read as "
        "ONE number — `complex(re, im)` — so an `int` (a `bool` included) "
        "at either position is stored as the `float` that number carries. "
        "`{'k':'complex','re':1,'im':0}` and the same document spelled "
        "`1.0`/`0.0` are therefore ONE document with ONE hash. No encoder "
        "produces the integer spelling (`_encode` writes `obj.real` and "
        "`obj.imag`, which are `float` whatever a complex was built from); "
        "the load face admits it because the number it denotes is the same "
        "one, and it is WRITTEN DOWN rather than refused for that reason. "
        "Audit 0.2.0 B12.",
    ),
    (
        "array payload spelling",
        "An `Array`'s `data` is recorded as base64 TEXT and stored as the "
        "BYTES it denotes, so two base64 spellings that denote one byte "
        "string are one document with one hash — base64's trailing bits are "
        "not part of the value it encodes, and neither python's "
        "`validate=True` nor `binascii`'s `strict_mode=True` treats them as "
        "part of it (measured: `'AB=='` and `'AC=='` both decode to "
        "`b'\\x00'`). `_encode` re-emits the canonical spelling. This is a "
        "property of the encoding rather than a choice this module made, "
        "and it is recorded because a canonicalization with no entry is the "
        "one thing no test here can catch. Audit 0.2.0 B12.",
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
# Tagged, recursive, and closed over exactly the types above. The content
# hash is defined over this encoding.
#
# ``to_dict`` / ``from_dict`` ROUND-TRIP LOSSLESSLY OVER THE OBJECTS THE
# LOAD DOOR ACCEPTS — which is not every object ``to_dict`` will encode,
# and this paragraph said it was until B12's own review. What is true, and
# is what the hash and every persisted verdict rest on:
#
#   * Every `ClosedJaxpr` `from_dict` RETURNS re-encodes and reloads to
#     itself with its `content_hash` preserved. That is not a corpus
#     result: the load rules are functions of the LOADED object, so
#     accepting a document is accepting its re-encoding.
#   * So does every query the TRACE FACE produces — measured, over the B12
#     census's 170-document legitimate population (every zero-argument
#     harness in the property corpus, the tag probes and the hand-built
#     bases, covering all 15 tags): 170 of 170 exact, hash preserved.
#
# WHAT IS NOT COVERED, and is each rule's stated scope arriving at the
# other face rather than a hole: TWO REFUSALS RUN ON THE LOAD PATH ONLY,
# so their subjects are CONSTRUCTIBLE AND NOT RELOADABLE.
# :func:`_validate_required_params` refuses a `JaxprEqn` missing a param
# jax supplies on every traced instance of its primitive, and hand-built
# IR legitimately omits params. :func:`_validate_decl_nonempty` refuses a
# `stelling_any` declaring an EMPTY set — ``(inf, inf)``, ``(nan, hi)``,
# ``lo > hi`` — which the suite builds THROUGH THE CONSTRUCTOR on purpose,
# over operands no `any_array` will produce; moving that rule to the
# constructor turns 11 pre-existing tests red across four files, measured
# and broken down in its own docstring. Both encode without complaint and
# are then refused by `from_dict`. Nothing is silent about it: the refusal
# is a `TranscriptionError`, it mints no verdict, and `content_hash` is
# computed from `to_dict` alone and is unaffected.
#
# `tests/test_document_schema.py` pins both halves and ENUMERATES the
# load-only rules from this file's own call graph rather than from this
# list, so a third one cannot arrive without that test going red and
# naming this paragraph. **WHAT "THE CALL GRAPH" MEANS THERE, because the
# first version of that instrument meant half of it:** the closure is
# seeded from `_decode` AND :func:`_validate_loaded`, which is the whole
# of `ClosedJaxpr.from_dict`. It was seeded from `_validate_loaded` alone
# until B12's own re-review, and a rule added to the DECODER was therefore
# invisible to it — five of this module's refusals live there. The
# decoder-side refusals are enumerated in their own bucket and do not
# widen this bound, because a `_doc_*` rule judges a DOCUMENT and never an
# object a constructor built. A rule the graph reaches from neither side —
# through an alias, a dispatch table or a lambda, which an `ast.Call` walk
# cannot follow — is caught by that test's THIRD assertion instead of
# being missed, and that one does not name this paragraph: it says the
# edge cannot be followed, which is a different thing to fix.
#
# WHY THE `isinstance` CHAINS BELOW ARE NOT THE ONES THE DOOR REPLACED,
# and the sweep's verdict on them, so the next reader does not have to
# re-derive it — audit 0.2.0 B6 audit 7.
#
# `_encode` walks an object graph that has ALREADY been through the
# canonicalization door: every value reachable from a constructed
# `ClosedJaxpr` is an exact built-in, an exact `ir` dataclass, or a
# registered type, so every `isinstance` here is answered by the object's
# REAL type — an exact object's ``__class__`` is its type. It also decides
# an ENCODING rather than gating a check, and it is the hot path of
# `content_hash`.
#
# `_decode` is the other direction and its input is whatever a caller
# hands `from_dict`. A leaf that lies about its type is returned unchanged
# from `_decode`'s scalar arm and then REFUSED by the door when a
# dataclass constructor receives it — for a value whose TYPE has no exact
# form to store, which is the question that door asks. **It is not the
# question "is this the type the encoding writes here", and that one was
# asked nowhere at all** — see the document-schema narrative below
# :func:`_encode`. What is NOT covered, and is recorded
# rather than repaired because it is a different surface with a different
# contract (this reader raises `ValueError`, not `TranscriptionError`, and
# its input is JSON): the READS OF THE MAPPING ITSELF. ``isinstance(obj,
# dict)`` takes a `__class__` property's word for it, and ``obj.get`` /
# ``obj[...]`` are the mapping's own — so `ClosedJaxpr.from_dict` raises a
# raw `AttributeError` on an object that merely claims to be a `dict`, and
# a raw `KeyError` on a real `dict` SUBCLASS whose ``get`` answers
# differently on two calls. Neither reaches a verdict: they are crashes in
# the reader, before any IR exists. The general repair is to route every
# read of the serialization through `dict`'s own accessors, which is a
# change to this reader's error surface and belongs in its own commit.


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


# --- the document's own shape, judged where the reader reads it -------------
#
# ONE RULE, TWO DOORS, AND THE SECOND DOOR IS WHERE THE ANNOTATIONS ALREADY
# ARE — audit 0.2.0 B12, S15 and S16's third and fourth items.
#
# :func:`_encode` above is a total function from IR to JSON with one arm per
# stored type, so the JSON type at every position of a document is not a
# convention anybody has to remember: it is what that function writes there.
# **Nothing asked.** Measured on `main` at `a4e4056`, from pure JSON through
# `ClosedJaxpr.from_dict`, no attacker Python anywhere in the document:
#
#     <eqn>.primitive  null / true / 0 / -1 / 1.5 / [] / [0]
#         all ACCEPTED, silently reclassified as an unknown primitive, and
#         the `stelling_assert` the equation carried DISAPPEARS — a REFUTED
#         two-obligation query returns VERIFIED with one obligation. That
#         is S15.
#     <tuple>.items  {}      ACCEPTED as the empty geometry ()
#     <tuple>.items  "xx"    ACCEPTED as ('x', 'x')
#     <tuple>.items  {"k":…} ACCEPTED as the dict's KEYS
#     <var>.id null, <aval>.weak_type "xx", <aval>.dtype 1.5, <dbg>.func 0
#         all ACCEPTED and round-tripped faithfully
#
# THE RULE IS THAT A DOCUMENT VALUE IS STORED ONLY WHERE ITS TYPE IS THE
# TYPE THE CODE DECLARES FOR ITS POSITION, and a position is one of exactly
# two kinds:
#
#   * a DATACLASS FIELD, whose declared type is its own annotation. That is
#     :func:`_spec_of`, applied by :func:`_canonicalise` to every field
#     of every dataclass in this module. The annotations are READ, from
#     ``typing.get_type_hints``, never listed — so a field added later is
#     covered without anyone remembering this comment, which is the whole
#     difference between this and a schema table that drifts.
#   * a SEQUENCE THIS READER MUST ITERATE in order to recurse, whose
#     declared type is the container `_encode` writes. That one cannot be an
#     annotation, because it is a fact about the ENCODING and not about the
#     stored value: `_decode` turns the document's `list` into the field's
#     `tuple`, so by the time a field exists the container is gone. It is
#     :func:`_doc_sequence`, which is :func:`_canonical_shell` — the
#     container reader this module already had, for the ``params`` sequence
#     alone — asked at every other sequence position instead of at one.
#
# WHAT THIS READER THEREFORE STILL HANDS OVER RAW, each because something
# else owns it and asks a STRONGER question:
#
#   * the two ``shape`` positions — :func:`_load_extents` reads every extent
#     through ``__index__`` and installs a plain `int`, which no type test
#     is as strong as (audit 0.2.0 B6 audit 8 put them here deliberately);
#   * the sequences with nothing to recurse into — ``effects``,
#     ``source_info``, ``arg_names``, ``result_paths``. This reader used to
#     call ``tuple()`` on them, which is the same second-reader mistake the
#     shapes had: ``tuple("xx")`` is ``('x','x')`` and ``tuple({})`` is
#     ``()``. Handing them over whole lets the ONE door judge the container
#     and the elements together, from the field's own ``tuple[str, ...]``;
#   * every scalar, judged the same way at the same door.
#
# AND THE THREE LEAVES THIS READER CONSUMES ITSELF, which have no field to
# be judged at because the field holds something else by then:
# ``<array>.data`` (base64 text here, `bytes` in the field) and
# ``<complex>.re``/``.im`` (two numbers here, one `complex` in the field).
# Those three are the only leaf positions with a type gate in this
# function, and :data:`CANONICALIZATIONS` records what each admits.
#
# WHY THE REFUSALS HERE ARE `TranscriptionError` WHEN THE THREE THIS READER
# ALREADY HAD ARE `ValueError`, stated rather than left to be noticed. The
# module's declared exception for "a document this module refuses" is
# `TranscriptionError`, every other refusal in the load pass raises it, and
# the residual class the B12 census measured names CATCHABILITY as its
# sharp half — 4,879 raw escapes of which ``except TranscriptionError``
# caught none. A new refusal that is not catchable through the declared
# type would be adding to that pile. The three older arms (unknown tag, a
# non-dict where a tagged object belongs, a document that is not a
# `ClosedJaxpr`) are a documented surface with tests pinned on their type
# and are left alone; so `from_dict` has two refusal shapes, which is
# reported in `SOUNDNESS.md` and not repaired here.


# THE CONTAINERS A DOCUMENT SEQUENCE MAY BE READ FROM, NAMED ONCE so a
# refusal can quote the rule instead of restating it. This is the pair
# :func:`_canonical_shell` accepts, and it is not a second implementation
# of that function: `tests/test_document_schema.py` COMPUTES the set that
# function actually accepts, over a population of container types it builds
# rather than lists, and requires it to be exactly this pair — so the name
# and the behaviour cannot drift apart in silence.
#
# `_encode` WRITES ONLY ONE OF THEM. A JSON document carries a `list` at
# every sequence position and never a `tuple`, so the pair is wider than
# the encoding by exactly the container a caller gets when they build the
# mapping in python by hand instead of loading it — the same route, and
# the same reason, :data:`_SHAPE_PARAM_CONTAINERS` is that pair for.
_DOC_SEQUENCE_CONTAINERS: tuple[type, ...] = (tuple, list)

# The rule as a sentence, DERIVED from the tuple above — the discipline
# :data:`_SHAPE_PARAM_RULE` and :data:`_CANONICAL_RULE` carry.
_DOC_SEQUENCE_RULE = " or ".join(
    f"a {t.__name__}" for t in _DOC_SEQUENCE_CONTAINERS
)

# The one sentence both complex-part gates quote, so the two positions
# cannot come to name different families.
_DOC_COMPLEX_PART_RULE = (
    "a complex value's real and imaginary parts are recorded as binary64, "
    "and this reader combines the two into one `complex` before any field "
    "exists to judge either"
)


def _doc_refuse(where: str, what: str) -> None:
    """:func:`_load_check`'s sibling for the DESERIALIZATION READER.

    `_load_check`'s message says *"refused at construction"* and points at
    ``__post_init__``, which is true of every other refusal in this pass
    and false of this one: nothing has been constructed yet, and ``where``
    names a position in the DOCUMENT rather than a field of a dataclass.
    Same exception type for the reason the narrative above gives.
    """
    raise TranscriptionError(
        f"malformed IR serialization: {where}: {what} — refused by "
        f"`ClosedJaxpr.from_dict`'s reader, before any IR object exists, so "
        f"this position is named as the document spells it and not as a "
        f"dataclass field (see the module docstring)"
    )


def _doc_sequence(v: object, where: str) -> tuple:
    """A document SEQUENCE position, as an EXACT tuple whose ITEMS are
    untouched — or refused, naming the position and the type it found.

    :func:`_canonical_shell` is the reader, unchanged and not copied: an
    exact ``tuple`` comes back as it stands, an exact ``list`` as the
    ``tuple`` of the same items, and a SUBCLASS of either is read through
    the base's own accessor — so a `list` subclass whose ``__iter__`` lies
    cannot hand this function a different sequence than the one it is
    holding. What this function adds is the POSITION, which that one has no
    way to know.
    """
    try:
        return _canonical_shell(v)
    except _NotCanonical as exc:
        _doc_refuse(
            where,
            f"value {_safe_repr(exc.obj)} is of type "
            f"{_safe_type_name(exc.obj)}, and this position is a SEQUENCE: "
            f"`_encode` writes a JSON list here and this reader takes "
            f"{_DOC_SEQUENCE_RULE}, and reading any "
            f"other container as one models something the document never "
            f"said (`tuple(\"xx\")` is `('x','x')`, `tuple({{}})` is `()` — "
            f"a geometry, an argument list or an equation list that came "
            f"out of thin air)",
        )
    raise AssertionError("unreachable")  # pragma: no cover


def _doc_leaf(v: object, types: tuple[type, ...], where: str, why: str) -> object:
    """A document LEAF this reader consumes ITSELF, judged by type.

    ``issubclass(type(v), types)`` and not ``isinstance``, for the reason
    :func:`_held_in_a_shape_param_container` gives: ``isinstance`` reads the
    OBJECT's ``__class__``, which a two-line property answers. A subclass is
    admitted rather than refused because the consumer below reads it ONCE
    (``complex(...)`` and ``base64.b64decode`` each take one read) and
    stores an exact value — the same bargain the canonicalization door
    makes.
    """
    if not issubclass(type(v), types):
        _doc_refuse(
            where,
            f"value {_safe_repr(v)} is of type {_safe_type_name(v)}, and "
            f"this position carries "
            + " or ".join(t.__name__ for t in types)
            + f": {why}",
        )
    return v


def _doc_complex(obj: dict, where: str) -> complex:
    """``{"k": "complex", "re": …, "im": …}`` as one `complex`, or refused.

    The construction is guarded as well as its inputs: ``complex(10**400,
    0.0)`` raises `OverflowError` for a document whose parts are both of an
    accepted type, and letting that out of `from_dict` raw is the same
    catchability finding this module already carries.
    """
    re = _doc_leaf(obj["re"], (int, float), f"{where}.re", _DOC_COMPLEX_PART_RULE)
    im = _doc_leaf(obj["im"], (int, float), f"{where}.im", _DOC_COMPLEX_PART_RULE)
    try:
        return complex(re, im)
    except Exception:  # noqa: BLE001 — unreadable IS the finding
        _doc_refuse(
            where,
            f"parts {_safe_repr(re)} and {_safe_repr(im)} are of an accepted "
            f"type but are not a binary64 pair, so there is no complex "
            f"number for this position to denote",
        )
    raise AssertionError("unreachable")  # pragma: no cover


def _doc_payload(v: object, where: str) -> bytes:
    """``<array>.data`` — the base64 TEXT `_encode` writes — as bytes.

    ``validate=True`` deliberately. The default SILENTLY DISCARDS every
    character outside the base64 alphabet, so a document saying
    ``"AAAA!!AAAAAAA="`` for an eight-byte array got the eight bytes of
    ``"AAAAAAAAAAA="`` instead — a payload it did not write, of exactly the
    right length, so the byte-length
    check in :func:`_validate_array_value` PASSED on it —
    the same *"reading it as something else models an array the document
    never described"* argument the shape rule makes, one position over. No
    encoder is affected: ``base64.b64encode`` emits nothing but the
    alphabet and its padding.
    """
    text = _doc_leaf(
        v, (str,), where, "an Array's payload is recorded as base64 text"
    )
    try:
        return base64.b64decode(text, validate=True)
    except Exception:  # noqa: BLE001 — undecodable IS the finding
        _doc_refuse(
            where,
            f"value {_safe_repr(v)} is not decodable base64, so the array "
            f"has no payload for its shape and dtype to describe",
        )
    raise AssertionError("unreachable")  # pragma: no cover


_CANONICAL_FIELDS: dict[type, tuple[str, ...]] = {}


def _field_names(cls: type) -> tuple[str, ...]:
    """The declared field names of an `ir` dataclass, cached per class.

    ONE READER FOR EVERY RULE THAT IS PER-FIELD — :func:`_canonicalise`
    (which field to canonicalize, and which spec to judge it against, since
    the spec table it builds is keyed by exactly these names) and
    :func:`_required_doc_keys` (which key the document must carry, for
    :func:`_doc_keys`) — so a field added to one of these dataclasses later
    is covered by all of them at once, which is a property none would have
    if any kept its own list.
    """
    names = _CANONICAL_FIELDS.get(cls)
    if names is None:
        names = _CANONICAL_FIELDS[cls] = tuple(
            f.name for f in _dataclasses.fields(cls)
        )
    return names


def _required_doc_keys(cls: type, optional: tuple[str, ...]) -> tuple[str, ...]:
    """:func:`_field_names` minus the fields ``to_dict(include_metadata=
    False)`` omits — which are exactly the two the hash-scope commitment in
    :data:`CANONICALIZATIONS` excludes."""
    return tuple(n for n in _field_names(cls) if n not in optional)


def _doc_keys(obj: dict, tag: str, required: tuple[str, ...],
              optional: tuple[str, ...] = ()) -> None:
    """The KEYS a tagged object carries: exactly the ones `_encode` writes.

    THE LAST OF THE READER'S RAW ESCAPES OVER THE B12 CENSUS SWEEP —
    audit 0.2.0 B12, and the scope is part of the claim. Every one of
    this reader's ``obj["…"]`` reads was unguarded, so DELETING any required
    key from a document produced a raw `KeyError` out of `from_dict`: 880 of
    the 20,424 cells of that sweep, and the only exception type
    left in that column once the schema rules above are in place. A
    `KeyError` is loud and cannot mint a verdict, so this is the robustness
    half of the finding rather than the soundness half — but
    ``except TranscriptionError`` catches none of it, which is what the
    census names as the sharp edge of the residual class.

    THE SWEEP IS SINGLE-POSITION AND FINITE-VALUED, AND ONE RAW ESCAPE
    LIVES OUTSIDE IT, unfixed by this rule and untouched by this batch:
    `_decode` recurses, so a ``{"k":"tuple","items":[…]}`` chain deep
    enough exhausts the interpreter stack and `from_dict` raises a bare
    `RecursionError`. It is reachable from PURE JSON — ``json.loads`` and
    ``json.dumps`` both accept the chain — and measured identically on
    `a4e4056` and here: depth 400 accepted and round-tripping, depth 900
    and 2000 a raw `RecursionError`, at the interpreter default limit of
    1000. Pre-existing, orthogonal to the key rule, and reported rather
    than repaired here because bounding a recursive reader's DEPTH is a
    different question with its own limit to choose.

    AND THE OTHER DIRECTION, which the census's single-position sweep could
    not reach because it only ever REPLACED a value that was already there:
    an UNKNOWN key was silently dropped. ``{"k":"var","id":0,"aval":…,
    "primitive":"stelling_assert"}`` loaded, re-encoded WITHOUT the extra
    key, and hashed as though the document had never said it — a document
    getting a different program than it wrote, which is the same sentence
    the container rule above is about.

    ``required`` IS THE DATACLASS'S OWN FIELD LIST, not a table: `_encode`
    writes ``"k"`` plus one key per field for all thirteen tagged
    dataclasses, and `tests/test_document_schema.py` COMPUTES that
    correspondence from `_encode`'s own output rather than trusting this
    sentence. ``optional`` is the two metadata fields
    ``to_dict(include_metadata=False)`` omits, which is the same list
    :data:`CANONICALIZATIONS`' hash-scope entry names.
    """
    have = set(obj)
    missing = sorted(k for k in required if k not in have)
    if missing:
        _doc_refuse(
            f"<{tag}>",
            f"the document is missing key(s) {missing} that `_encode` writes "
            f"on every {tag!r}: a tagged object without them is not a "
            f"serialization of one, and this reader has no value to put "
            f"there — before this check it raised a bare `KeyError` from "
            f"the middle of the read, which `except TranscriptionError` "
            f"does not catch",
        )
    extra = sorted(have - {"k"} - set(required) - set(optional))
    if extra:
        _doc_refuse(
            f"<{tag}>",
            f"the document carries key(s) {extra} that `_encode` never "
            f"writes on a {tag!r}: this reader would drop them, so the "
            f"document would round-trip to something it does not say and "
            f"hash as though it had never said it",
        )


def _doc_pair_entry(entry: object, where: str) -> tuple:
    """One ``[key, value]`` entry of a ``params`` / ``fields`` sequence.

    **The CONTAINER is this reader's question and the PAIR-NESS is not**,
    which is the difference between a rule stated once and a rule stated
    twice. `for key, value in <entry>` unpacks a 2-character `str` into a
    key and a value the document never wrote, so the container has to be
    judged here, before the unpack; but whether there are exactly two of
    them, and whether the key is a name, is asked by
    :func:`_canonical_param_keys` and by ``NamedTupleParam``'s own field
    annotation — both of which are reachable without this reader and must
    therefore ask anyway. So an entry of the wrong LENGTH is returned as it
    stands, with its value undecoded, for whichever of those two refuses it.
    """
    items = _doc_sequence(entry, where)
    if len(items) != 2:
        return items
    return (items[0], _decode(items[1]))


def _decode(obj: object) -> object:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if not isinstance(obj, dict):
        raise ValueError(
            f"malformed IR serialization: unexpected {_safe_type_name(obj)}"
        )
    k = obj.get("k")
    if k == "complex":
        # the two tags with no dataclass behind them name their keys here;
        # the thirteen that have one read the field list off the class
        _doc_keys(obj, "complex", ("re", "im"))
        return _doc_complex(obj, "<complex>")
    if k == "tuple":
        _doc_keys(obj, "tuple", ("items",))
        return tuple(
            _decode(x) for x in _doc_sequence(obj["items"], "<tuple>.items")
        )
    # THE `shape` IS HANDED OVER RAW, and that is the repair rather than
    # an omission — audit 0.2.0 B6 audit 8. Both of these read
    # ``tuple(obj["shape"])``, which is a SECOND reader of the document's
    # shape standing in FRONT of the one that owns the question: it ran
    # before any `__post_init__`, so `2` and `null` came out of
    # `from_dict` as a raw ``TypeError: 'int' object is not iterable``
    # (which `except TranscriptionError` does not catch, because
    # `TranscriptionError` SUBCLASSES `TypeError`), and `{}` came out
    # ACCEPTED as the scalar shape `()` because a dict iterates to
    # nothing. `_load_extents` now judges the container, the iteration and
    # each extent, and `Aval.__post_init__` / `Array.__post_init__`
    # install what it read — so a `list` from JSON still becomes a stored
    # `tuple`, by the reader that checked it instead of by one that did
    # not.
    if k == "aval":
        _doc_keys(obj, "aval", _field_names(Aval))
        return Aval(
            kind=obj["kind"],
            shape=obj["shape"],
            dtype=obj["dtype"],
            weak_type=obj["weak_type"],
        )
    if k == "array":
        _doc_keys(obj, "array", _field_names(Array))
        return Array(
            dtype=obj["dtype"],
            shape=obj["shape"],
            data=_doc_payload(obj["data"], "<array>.data"),
        )
    if k == "var":
        _doc_keys(obj, "var", _field_names(Var))
        return Var(id=obj["id"], aval=_decode(obj["aval"]))
    if k == "lit":
        _doc_keys(obj, "lit", _field_names(Literal))
        return Literal(val=_decode(obj["val"]), aval=_decode(obj["aval"]))
    if k == "enum":
        _doc_keys(obj, "enum", _field_names(EnumParam))
        return EnumParam(cls=obj["cls"], member=obj["member"])
    if k == "sentinel":
        _doc_keys(obj, "sentinel", _field_names(SentinelParam))
        return SentinelParam(cls=obj["cls"])
    if k == "opaque":
        _doc_keys(obj, "opaque", _field_names(OpaqueParam))
        return OpaqueParam(cls=obj["cls"])
    if k == "treedef":
        _doc_keys(obj, "treedef", _field_names(TreeDefParam))
        return TreeDefParam(text=obj["text"])
    if k == "ntuple":
        _doc_keys(obj, "ntuple", _field_names(NamedTupleParam))
        return NamedTupleParam(
            cls=obj["cls"],
            fields=tuple(
                _doc_pair_entry(e, f"<ntuple>.fields[{i}]")
                for i, e in enumerate(
                    _doc_sequence(obj["fields"], "<ntuple>.fields")
                )
            ),
        )
    if k == "eqn":
        # ``effects`` and ``source_info`` GO OVER WHOLE: they hold no tagged
        # object, so this reader has nothing to recurse into and no reason to
        # be the one that judges their container — `JaxprEqn`'s own
        # ``tuple[str, ...]`` annotation judges the container and every
        # element at one door. `tuple(obj["effects"])` here was the second
        # reader: it read `"ab"` as two effects and `{}` as none.
        _doc_keys(obj, "eqn", _required_doc_keys(JaxprEqn, ("source_info",)),
                  ("source_info",))
        return JaxprEqn(
            primitive=obj["primitive"],
            invars=tuple(
                _decode(a) for a in _doc_sequence(obj["invars"], "<eqn>.invars")
            ),
            outvars=tuple(
                _decode(v) for v in _doc_sequence(obj["outvars"], "<eqn>.outvars")
            ),
            params=tuple(
                _doc_pair_entry(e, f"<eqn>.params[{i}]")
                for i, e in enumerate(_doc_sequence(obj["params"], "<eqn>.params"))
            ),
            effects=obj["effects"],
            source_info=obj.get("source_info", ()),
        )
    if k == "dbg":
        _doc_keys(obj, "dbg", _field_names(DebugInfo))
        return DebugInfo(
            func=obj["func"],
            arg_names=obj["arg_names"],
            result_paths=obj["result_paths"],
        )
    if k == "jaxpr":
        # ``is not None`` AND NOT A TRUTH TEST — audit 0.2.0 B12. `_encode`
        # writes `null` when and only when there is no `DebugInfo`, so
        # ``if dbg`` asked the document a question the encoding does not
        # answer: `0`, `""`, `[]`, `false` and `{}` were each read as "no
        # debug info", which is a document silently getting a different
        # answer than it wrote rather than a refusal it can be told about. A
        # `DebugInfo` is never falsy (it defines no `__bool__` and no
        # `__len__`), so no legitimate document moves.
        _doc_keys(obj, "jaxpr", _required_doc_keys(Jaxpr, ("debug_info",)),
                  ("debug_info",))
        dbg = obj.get("debug_info")
        return Jaxpr(
            constvars=tuple(
                _decode(v)
                for v in _doc_sequence(obj["constvars"], "<jaxpr>.constvars")
            ),
            invars=tuple(
                _decode(v) for v in _doc_sequence(obj["invars"], "<jaxpr>.invars")
            ),
            outvars=tuple(
                _decode(a) for a in _doc_sequence(obj["outvars"], "<jaxpr>.outvars")
            ),
            eqns=tuple(
                _decode(e) for e in _doc_sequence(obj["eqns"], "<jaxpr>.eqns")
            ),
            effects=obj["effects"],
            debug_info=_decode(dbg) if dbg is not None else None,
        )
    if k == "closed":
        _doc_keys(obj, "closed", _field_names(ClosedJaxpr))
        return ClosedJaxpr(
            jaxpr=_decode(obj["jaxpr"]),
            consts=tuple(
                _decode(c) for c in _doc_sequence(obj["consts"], "<closed>.consts")
            ),
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


def _held_in_a_shape_param_container(obj: object) -> bool:
    """Is ``obj`` held in one of :data:`_SHAPE_PARAM_CONTAINERS`?

    ONE OBJECT WAS NOT ENOUGH: THE TWO FACES MUST ALSO ASK IT THE SAME WAY
    — audit 0.2.0 B6 audit 7. :func:`_validate_decl_eqn` and
    :meth:`stelling.obligation._Slicer._declared_shape` both asked
    ``isinstance`` against :data:`_SHAPE_PARAM_CONTAINERS`, which shares
    the list but not the READING, and a reading is a thing two faces can
    drift apart in exactly as a list is. They call this now, so neither
    can be hardened without the other.

    AND THE READING IS ``issubclass(type(obj), ...)``. ``isinstance``
    falls back to the OBJECT's ``__class__``, so a two-line property
    returning ``tuple`` satisfied it from any object at all — and the
    sentence that is supposed to refuse such a param then never fired, on
    either face. ``issubclass`` dispatches ``type.__subclasscheck__`` on
    the BASE — both bases here have metaclass exactly ``type``, no
    ``ABCMeta`` in play — so the derived class does not get a say.
    """
    return issubclass(type(obj), _SHAPE_PARAM_CONTAINERS)


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
    a per-QUOTE count (the ten above) is larger again. The four
    measurements, each computed by `tests/test_ir_message_totality.py`
    rather than trusted from this paragraph:

        this tree, as shipped              95 swept / 0 escapes / 20 skipped
        guards neutered, door shipped       1 escape  /  1 line  / 1 message
        guards neutered, and the door's
          LEAF READS neutered too          87 escapes /  5 lines / 5 messages
        ... and the FIELD-ANNOTATION rule
          neutered as well                 29 escapes / 10 lines / 10 messages
        `30d4b04`, guards absent           28 escapes / 10 lines / 8 messages

    THE THIRD ROW IS NO LONGER THE DEEPEST MEASUREMENT, and the fourth is
    why it was added — audit 0.2.0 B12. :func:`_canonicalise` refuses a
    field whose value is not of the type the field DECLARES, and that
    refusal stands IN FRONT OF six of the quote sites below it: with it
    shipped a hostile leaf is refused at its one message expression and
    the six deeper ones are never composed, so the third row's per-MESSAGE
    figure FELL from 9 to 5 while its escape COUNT rose from 27 to 87.
    Neither movement is a weakening and neither is visible in one row,
    which is the silent shrinkage the two-knob control was built to avoid,
    arriving through a fix.

    The union the record quotes is therefore taken over CONFIGURATIONS and
    not over the deepest one: the 11 message expressions the sweep reaches
    in one configuration or the other, plus the 2 the canonical document
    masks, comes to 13, which is the union that test computes. The ten
    QUOTES above is a different unit and a fixed statement about the tree
    audit 4 measured; the two agreed at ten on that tree and no longer do,
    and re-typing the second number to keep the coincidence alive would be
    this file's own defect one step subtler. A guarded quote is cheap;
    deciding per site which objects can misbehave is how those six were
    missed.

    **THE GUARD-NEUTERED FIGURES ARE MEASURED WITH THE CANONICALIZATION
    DOOR REMOVED TOO** — audit 0.2.0 B6 audit 6. Every hostile leaf that
    sweep injects is a SUBCLASS of a stored type, and the door replaces
    one with an exact twin before any message quotes it, so with the door
    shipped all but one of the deepest row's escapes do not happen. A
    control that neutered
    only the guards measured 1 escape and would have gone on passing with
    every call to this function deleted, so it neuters the door's leaf
    reads as well and reports each measurement with the other out of the
    way. **AND THE FIELD-ANNOTATION RULE IS A THIRD DEFENCE with the same
    property** (audit 0.2.0 B12), so there is a third knob and a fourth
    row; see the table above for what each removes and why the union is
    taken over configurations.

    **AND "1" IS NOT A COUNT OF THE SITES THIS FUNCTION IS LOAD-BEARING
    AT** — audit 0.2.0 B6 audit 7, which is how the paragraph above was
    read, and how it was written: *"they are not guarded — they are
    unreachable"*. Unreachable TO THE LEAF THAT SWEEP INJECTS, which is a
    subclass, and only that. A leaf that BYPASSES the door — the metaclass
    and `__class__` liars audit 7 built, before the door was fixed to
    refuse them — reaches every one of the sites the DEEPEST row above
    counts, carrying whatever
    `__repr__` it likes, and the guards are then the only thing between it
    and a raw crash out of a public constructor. So this function is
    load-bearing at all of them, not at 1. **THE FIGURE IS NAMED BY ITS
    ROW AND NOT TYPED HERE** — audit 0.2.0 B12, because it was typed here
    as `27`, which was the third row's escape count when this paragraph
    was written and is the FOURTH row's now that a third defence exists.
    A digit in prose that nothing recomputes is this file's own recurring
    defect, and a digit that silently means a different row is that defect
    with the sentence still reading true. What "1" measures is the residue of
    ONE sweep against ONE defence, and the site it names is real and
    worth naming: the hostile extent inside a declaration's `shape` param,
    which `_validate_decl_eqn` reads as handed in because it has its own
    rule for that param."""
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
    `Aval(...)` and `JaxprEqn(...)` raw (audit 0.2.0 B6 audit 3, F2).

    **AND THE ITERATION ITSELF WAS UNGUARDED, WHICH IS THE SAME FINDING
    ONE LEVEL OUT** — audit 0.2.0 B6 audit 8. The per-extent `try`
    described above covered `operator.index(d)` and not the `for d in
    shape` that produced `d`, so a `shape` that is not a
    sequence at all left the same public constructors raw, and — worse —
    a `shape` that iterates to NOTHING was accepted as the scalar shape
    `()`. Measured AT `ac2dcb1`, and identically on `dff95fc` and on
    `main` at `198a2b5`, from PURE JSON through `ClosedJaxpr.from_dict`
    with no Python object anywhere in the document — all five rows are
    `TranscriptionError` on this tree:

        aval shape 2       raw `TypeError: 'int' object is not iterable`
        aval shape null    raw `TypeError: 'NoneType' object ...`
        aval shape 1.5     raw `TypeError: 'float' object ...`
        aval shape true    raw `TypeError: 'bool' object ...`
        aval shape {}      ACCEPTED, and the aval records shape ()

    That last row is the one that matters: `{}` is not a malformed extent
    the caller can be told about, it is a document that got a DIFFERENT
    array than it wrote, silently, and every later reader — the slicer,
    the propagator's element counts, the emission — then models a scalar
    where the document said nothing of the kind. The others are the
    catchability finding this module already carries: `TranscriptionError`
    SUBCLASSES `TypeError`, so `except TranscriptionError` does not catch
    a raw `TypeError`, and unlike the three routes recorded in
    `tests/test_canonicalization_routes.py` this one needs no attacker
    Python — a JSON file reaches it.

    **THE RULE IS THE ONE THIS MODULE ALREADY STATES, applied to the one
    shape it was not applied to.** :func:`_validate_decl_eqn` has judged a
    declaration's `shape` PARAM by :data:`_SHAPE_PARAM_CONTAINERS` since
    audit 4, for a reason that is about the aval's shape word for word —
    *reading any other container as a shape models an array the document
    never described* (`tuple(b"34")` is `(51, 52)`; `tuple({})` is `()`).
    An aval and an array shape are now asked the same question by the
    same predicate, so the two cannot be hardened apart, and the reading
    is `issubclass(type(obj), ...)` for the reason
    :func:`_held_in_a_shape_param_container` gives.

    WHAT THAT NARROWING COSTS, stated rather than assumed: every route
    that is not hand-built IR already arrives in one of those two
    containers — `_decode` builds a `list` from JSON, `_jax_compat.
    any_array` normalises with `tuple(int(d) for d in shape)`, and a jax
    or numpy `.shape` is a `tuple` — so nothing that reaches this function
    through a document or a trace changes hands. It is the PER-EXTENT rule
    that stays deliberately wide: an extent may still be any object with a
    working `__index__` (a numpy integer, one of the suite's
    `_TwoFacedExtent`s), because that is a claim about the extent and this
    is a claim about the container."""
    # THE CONTAINER, THEN THE ITERATION, THEN THE EXTENTS — three
    # different facts about the document, refused separately because a
    # single sentence for all three tells a reader holding `{}` that it
    # has a "non-integer shape extent", which it does not have and is not
    # why it was refused (the split `_validate_decl_eqn` makes, and for
    # the same reason).
    if not _held_in_a_shape_param_container(shape):
        return (
            f"a shape {_safe_repr(shape)} of type {_safe_type_name(shape)}: "
            f"a shape is recorded in {_SHAPE_PARAM_RULE}, and reading any "
            f"other container as one would model an array the document "
            f"never described (`tuple(b\"34\")` is `(51, 52)`, `tuple({{}})` "
            f"is `()`)",
            (),
        )
    try:
        items = tuple(shape)
    except Exception:  # noqa: BLE001 — unreadable IS the finding
        # `issubclass` is a claim about the TYPE and not about the object:
        # a `list` SUBCLASS whose `__iter__` raises satisfies the test
        # above, and letting that out of a public constructor raw is the
        # class this function is closing.
        return (
            f"a shape in {_SHAPE_PARAM_RULE} whose iteration RAISES, so its "
            f"extents cannot be read at all",
            (),
        )
    out = []
    for d in items:
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
# by an EXACT instance of a type this module is closed over — or, for the
# two kinds of object that are carried rather than rewritten, by an object
# whose single-valuedness some other invariant owns and states (an `ir`
# frozen dataclass, which canonicalized its own fields; a
# :data:`_LIBRARY_STORED_TYPES` registration, which delegates in writing).
# For a value in the first class a later read cannot differ, because there
# is no subclass left to answer it; for the other two the claim is the one
# the invariant makes and no wider. That is a property of the STORED
# OBJECT rather than of any reader, so it holds for readers nobody has
# written yet, which is what the per-member repairs could not do.
#
# AND THE DOOR'S OWN DISPATCH IS PART OF WHAT IS STORED — audit 0.2.0 B6
# audit 7, and the reason the paragraph above was false when it was
# written. Every sentence here is about what happens AFTER the door
# decides to read; the decision itself was made with the two most
# overridable tests in Python. ``type(obj) in <a frozenset of types>``
# runs the METACLASS's ``__hash__``/``__eq__``, so three lines of
# metaclass answering as ``float`` walked an arbitrary object out of the
# door untouched before any accessor ran; ``isinstance(obj, base)`` falls
# back to reading ``obj.__class__``, so a two-line property returning
# ``bool`` reached an arm whose read was the IDENTITY and stored the
# object unchanged.
#
# EACH BYPASS ALONE LEFT A TWO-FACED OBJECT STORED, and they are measured
# separately because a repair that closed only their conjunction would
# leave both open: the metaclass alone stores for ANY face, and the
# ``__class__`` property alone stores for ``bool``, the one face whose arm
# was the identity (for every other face it reached a real accessor and
# raw-crashed instead — see :func:`_read_or_refuse`). Driven together they
# minted a `discharged` on two obligations that cannot both hold —
# ``sum(x) <= C`` and ``C <= 79/20`` over ``[1,2]^2``, whose true maximum
# is 4 — with every object in the document built through a public
# `stelling.ir` dataclass and no ``object.__setattr__`` anywhere, and they
# mint it identically on `main` and on the released `v0.1.0`. So the door
# asks, and may only ask:
#
#   * ``type(obj)`` — the object's header, which no `__class__` property
#     can move (``isinstance`` reads the property; ``type()`` does not);
#   * IDENTITY against the types it stores, never ``==``/``hash``, which
#     is :data:`_STORED_AS_IS` below — one ``id()`` and one ``dict``
#     lookup on the exact ``int`` it returns, protocols no document-
#     supplied object takes part in;
#   * ``issubclass(type(obj), base)``, which dispatches
#     ``type.__subclasscheck__`` ON THE BASE — a built-in whose metaclass
#     is exactly ``type`` — and so cannot be redirected by the derived
#     class at all. See :func:`_read_or_refuse` for the one thing a
#     derived class CAN still do to it, and why the accessor detects it.
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
# values are exact already, pay one ``id()`` and one ``dict`` lookup per
# value and allocate nothing.
#
# AND NO ARM'S READ MAY BE THE IDENTITY — audit 0.2.0 B6 audit 7. The
# table below carried ``(bool, lambda v: v)``, justified by "``bool``
# cannot itself be subclassed, so its read is the identity". The premise
# is true and the conclusion does not follow, for two measured reasons.
# The arm was entered by ``isinstance``, which asks the OBJECT what its
# class is, so a `__class__` property returning ``bool`` reached it from
# any object at all. And even with the entry test fixed, ``bool`` shares
# ``int``'s instance layout, so a REAL ``int`` subclass whose metaclass
# overrides ``mro()`` to return ``[cls, bool, int, object]`` is created
# without complaint and satisfies ``issubclass(cls, bool)`` — measured.
#
# THE SCOPE OF THAT SENTENCE IS THE ARMS THAT READ ``issubclass``, AND IT
# WAS WRITTEN AS IF IT WERE EVERY TYPE THIS MODULE STORES — audit 0.2.0 B6
# audit 8, which is this module's own recurring defect one more time. It
# said *"``bool`` is the only type this module stores for which it is
# possible"*. Measured on this tree (python 3.12.3, 21 stored types):
# **16 of the 21 can have an MRO entry forged for them** — the 13 `ir`
# dataclasses, `NoneType` and `IntervalArray`, whose solid base is
# ``object`` so a plain heap class adds no conflicting layout, plus
# ``bool`` from a real ``int`` subclass. What CPython's layout check
# refuses is the 5 that carry their own instance layout: ``int``,
# ``float``, ``complex``, ``str`` and ``bytes``.
#
# It is nonetheless not a hole, and the reason is the door's shape rather
# than CPython's: FORGING AN MRO MOVES ONLY ``issubclass``, and every base
# this module dispatches ``issubclass`` against is one CPython refuses to
# forge — the five :data:`_CANONICAL_READS` bases above, and ``tuple`` and
# ``list`` from :data:`_SHAPE_PARAM_CONTAINERS`, which carry their own
# instance layouts too. Membership in what this module STORES is decided
# by ``id(type(obj))`` against
# :data:`_STORED_AS_IS`, which is identity against the real type object
# and which a forged MRO does not touch. Measured: a forged claimer of
# each of the 13 `ir` dataclasses, of `NoneType` and of `IntervalArray`,
# instantiated with ``object.__new__`` so the base's ``__post_init__``
# never runs, was refused by :func:`_canonical` — 15 of 15,
# ``_NotCanonical`` every time. :func:`_read_or_refuse`'s own docstring
# has always stated the claim in this scope; this paragraph did not, and
# `tests/test_canonicalization_routes.py` now computes both halves.
#
# Under the identity read that object was STORED; under ``int.__index__``
# it is stored as the exact ``int`` it actually carries. An identity read
# is the one read that cannot detect having been applied to the wrong
# object, so the fix is not a better guard in front of it: it is that
# there is no identity read. ``bool`` keeps its place in what may be
# stored through :data:`_CANONICAL_UNSUBCLASSABLE`, which is a statement
# about EXACT types and reads nothing.

import dataclasses as _dataclasses  # noqa: E402  (stdlib; kept local to the pass)
import types as _types  # noqa: E402  (stdlib; for `X | Y` annotations)
import typing as _typing  # noqa: E402  (stdlib; reads the field annotations)


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


# THE TYPES WITH NO READ, BECAUSE THEY HAVE NO INEXACT FORM. CPython
# refuses both as a base class (``type 'NoneType' is not an acceptable
# base type``, and the same for ``bool``), so an object whose ``type()``
# is one of these is EXACT by construction and an object that merely
# CLAIMS to be one is a liar no read of it could settle. That is why
# neither appears in the read table below and why neither needs to:
# `_canonical` settles both by identity, before any read. The premise is
# TESTED rather than asserted — `tests/test_ir_canonicalization.py`
# subclasses each of them and requires the refusal — so a type added here
# whose subclassing CPython permits does not silently acquire an
# identity read (audit 0.2.0 B6 audit 7).
_CANONICAL_UNSUBCLASSABLE: tuple[type, ...] = (type(None), bool)

# THE READS, ONE PER BASE TYPE, IN THIS ORDER, AND NOT ONE OF THEM IS THE
# IDENTITY (see the door's narrative above). ``bool`` is an ``int``
# subclass and must be settled before ``int`` is asked, or
# ``int.__index__`` would store ``True`` as ``1`` and change what a param
# says — and it IS settled before, by ``_CANONICAL_UNSUBCLASSABLE``, which
# `_canonical` consults before it reaches this table at all.
_CANONICAL_READS: tuple[tuple[type, object], ...] = (
    (int, int.__index__),
    (float, float.__float__),
    (complex, lambda v: complex(complex.real.__get__(v),
                                complex.imag.__get__(v))),
    (str, str.__str__),
    (bytes, lambda v: bytes.__getitem__(v, slice(None))),
)

# The leaf types that are already what they will be stored as when their
# type is EXACT. Derived from the two tables above so the three cannot
# name different sets. ORDERED, because :data:`_CANONICAL_RULE` is the
# same list read as a sentence and a refusal that reorders itself between
# runs is a refusal no test can hold.
_CANONICAL_EXACT_TYPES: tuple[type, ...] = (
    _CANONICAL_UNSUBCLASSABLE + tuple(t for t, _ in _CANONICAL_READS)
)
_CANONICAL_EXACT: frozenset[type] = frozenset(_CANONICAL_EXACT_TYPES)

# The rule as a sentence, DERIVED from the table, so a refusal can never
# name a different set from the one it applied — the same discipline
# :data:`_SHAPE_PARAM_RULE` carries. ``_CANONICAL_IR_TYPES`` is computed
# at the foot of this module from the module's own dataclasses.
_CANONICAL_RULE = (
    ", ".join(
        ["None" if t is type(None) else t.__name__
         for t in _CANONICAL_EXACT_TYPES]
        + ["tuple"]
    )
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
# own `__post_init__` validates it). NO DOCUMENT CAN REACH THIS ARM, and
# the whole reason is `_decode`: it has no tag for such a type, so no JSON
# document produces one and a registered value can only come from a caller
# who built it in this process.
#
# **NOT `_encode`, AND NOT `to_dict` OR `content_hash` EITHER — this
# paragraph named all three, and it is the third copy of that claim in
# this file: the two below were corrected first, and this one — the one a
# would-be registrant reads before either of them — was found only by
# B12's re-review of its own fix.** `_encode` refuses a registered value
# only in the arms where it RECURSES: a `ClosedJaxpr` whose ``consts``
# hold one does raise ``stelling.ir cannot encode IntervalArray``, and
# that is the arm the sentence was generalising from. At the EIGHTEEN
# slots it writes STRAIGHT THROUGH, ``to_dict()`` returns a dict with the
# `IntervalArray` sitting in it and raises nothing — so a registered value
# is not outside `to_dict` at all. `content_hash` raises at fourteen of
# those eighteen (from ``json.dumps``, not from this module) and ANSWERS
# at the other four: ``<eqn>.source_info[*]`` and the three ``<dbg>``
# slots, which are exactly the metadata ``to_dict(include_metadata=False)``
# drops — so the hash is a correct function of a scope that deliberately
# excludes them, and not a refusal. All measured by driving every
# position; see the door narrative below :func:`_encode` for the
# enumeration and `tests/test_document_schema.py` for the driver.
#
# AND WHAT REGISTRATION IS *NOT* A DEFENCE AGAINST — audit 0.2.0 B6 audit
# 7. :func:`_register_stored_type` is module-level and importable, so any
# code running in this process can opt a type out of the door. That is
# checked below for the one property the delegation actually rests on,
# and it is still not a security boundary, because nothing at this level
# could be: code that can call it can equally rebind :func:`_canonical`.
# The boundary this door defends is a DOCUMENT — the values that arrive
# through `from_dict`, through a trace, or through a public dataclass
# constructor — and no document reaches this arm at all, for the reason
# the paragraph above gives. The check is here so that a future in-tree
# registrant cannot delegate single-valuedness to a class that does not
# have it; the disclosure is here so the check is not mistaken for more.
_LIBRARY_STORED_TYPES: tuple[type, ...] = ()


def _register_stored_type(cls: type) -> type:
    """Declare ``cls`` a library-supplied type :func:`_canonical` carries.

    Package-internal (see the comment above for what it delegates and for
    what this check is and is not). Returns ``cls`` so it can be used as a
    decorator.

    WHAT THE CHECK BELOW ESTABLISHES, AND WHAT IT DOES NOT — audit 0.2.0
    B6 audit 8, which read the previous wording against the code and found
    it claiming the conclusion from a premise that does not carry it. It
    said the frozen test made single-valuedness *"true because it is a
    FROZEN dataclass"*, and that *"a plain class with a property for a
    field ... would be exactly the hole the door exists to close"* — while
    this function ACCEPTS precisely that class.

    ESTABLISHED: ``cls`` is a frozen dataclass, so ``setattr`` on an
    instance raises and no later assignment can REBIND a field. That is a
    floor and it is worth having; it is not the property the delegation
    needs.

    NOT ESTABLISHED: single-valuedness. Frozen-ness stops rebinding, not
    TWO-FACEDNESS, and the two are different because a field name can
    resolve to a class-level DESCRIPTOR that was never an instance
    attribute at all. Measured on this tree: a frozen dataclass that
    DECLARES a field and gives that same name a ``property`` in its class
    body — `dataclasses.fields` lists the field — passes every test below,
    and
    :func:`_canonical` then carries it while it answers ``(1.0, 1.0)``
    on one read and ``(99.0, 99.0)`` on the next. There is no test here
    that would catch that, and this docstring no longer says there is.

    WHAT BOUNDS IT INSTEAD, all three measured rather than argued:

    * NO DOCUMENT REACHES THIS ARM, and the whole reason is `_decode`: it
      has no tag for a registered type (``unknown tag 'interval'``), so no
      JSON document can produce one and a registered value can only come
      from a caller who built it in this process. **NOT `_encode`, which
      this paragraph also named until B12's own review.** `_encode`
      refuses a registered value (``stelling.ir cannot encode
      IntervalArray``) only in the arms where it RECURSES — a `ClosedJaxpr`
      whose ``consts`` hold one does raise out of `to_dict`, correctly.
      At the slots it writes STRAIGHT THROUGH, ``to_dict()`` returns a
      dict with the `IntervalArray` sitting in it and raises nothing:
      **18 such positions, measured, not read off this list** — see the
      door narrative below :func:`_encode` for the enumeration and for
      the same correction at the other site. `content_hash` raises on such
      an object at FOURTEEN of the eighteen — from `json.dumps` rather
      than from this module, which is loud but is not this file refusing
      anything — and ANSWERS at the other four: ``<eqn>.source_info[*]``,
      ``<dbg>.func``, ``<dbg>.arg_names[*]``, ``<dbg>.result_paths[*]``.
      Those four are exactly the metadata
      ``to_dict(include_metadata=False)`` omits, and `content_hash` is
      defined over that reduced document, so the hash it returns is a
      correct function of a scope that deliberately excludes them. No
      soundness consequence, and stated because *"`content_hash` does
      still raise"* — unqualified, as this sentence read until B12's own
      re-review — is false at four of the eighteen positions the
      paragraph is about.
    * A SUBCLASS OF A REGISTERED TYPE IS REFUSED, not carried: membership
      is ``id(type(obj))`` against :data:`_STORED_AS_IS`, which is the
      registered type itself, so the property-backed variant has to be
      the type someone REGISTERED — it cannot be smuggled in under an
      existing registration. Measured: an `IntervalArray` subclass with a
      ``los`` property is refused as ``_NotCanonical``.
    * AND THIS FUNCTION IS NOT A SECURITY BOUNDARY, for the reason the
      comment above it gives: code that can call it can equally rebind
      :func:`_canonical`. It exists so a future IN-TREE registrant states
      its claim deliberately, and the claim it must state is
      single-valuedness — which this check does not verify for it.

    `tests/test_canonicalization_routes.py` pins the accepted
    property-backed field, so the gap is a measurement and not a sentence.

    Raises a plain `TypeError` and not a `TranscriptionError`: nothing
    here is a malformed document, and a caller catching document errors
    should not swallow a package-internal contract violation.
    """
    global _LIBRARY_STORED_TYPES
    if not issubclass(type(cls), type):
        raise TypeError(
            f"_register_stored_type takes a class; got {_safe_type_name(cls)}"
        )
    # the quotes go through `_safe_repr` like every other object-valued
    # quote in this pass: `cls.__name__` is an attribute lookup on a
    # caller-supplied class and a metaclass can make it refuse
    if id(cls) in _STORED_AS_IS:
        raise TypeError(
            f"{_safe_repr(cls)} is already a type this module stores, so "
            f"registering it would move which arm of the door decides it"
        )
    if not (_dataclasses.is_dataclass(cls)
            and getattr(cls, "__dataclass_params__", None) is not None
            and cls.__dataclass_params__.frozen):
        raise TypeError(
            f"{_safe_repr(cls)} is not a frozen dataclass: a registered type "
            f"is CARRIED rather than canonicalized, and the registering "
            f"module's claim to own its single-valuedness rests on its fields "
            f"being settled at construction and unassignable afterwards"
        )
    if not any(cls is k for k in _LIBRARY_STORED_TYPES):
        _LIBRARY_STORED_TYPES = _LIBRARY_STORED_TYPES + (cls,)
        _rebuild_stored_index()
    return cls


# THE DOOR'S MEMBERSHIP TEST, AND THE WHOLE OF WHY IT IS SPELLED THIS WAY
# — audit 0.2.0 B6 audit 7. Every type in it is stored AS IS: an exact
# leaf built-in, one of this module's own frozen dataclasses (canonical by
# its own ``__post_init__``), or a registered library type (carried, by
# the declaration above). Three arms, one answer, so one lookup.
#
# KEYED BY ``id()``, NOT BY THE TYPE. ``t in <set of types>`` and
# ``t == k`` both run the METACLASS, which a document-supplied object
# chooses; ``id()`` has no override hook at all, and the key it returns is
# an exact ``int`` whose ``hash`` and ``eq`` are ``int``'s own. The dict
# holds each type as its VALUE as well, which is not decoration: an
# address may only be reused after the object at it is freed, and holding
# the reference is what makes that impossible.
#
# WHY NOT ``any(t is k for k in ...)``, which is equally unforgeable.
#
# EVERY FIGURE BELOW IS AN OFF-TREE MEASUREMENT, dated and attributed, and
# no control computes any of it: a microbenchmark asserted in a suite is a
# promise about someone else's machine. Measured 2026-08-17, python
# 3.12.3, `/home/nick/venvs/stelling-jax`, 200k iterations per cell, three
# fresh processes, over a 20-type set:
#
#   ``t in <frozenset>``            23-39 ns    FORGEABLE (the metaclass)
#   ``id(t)`` + one dict lookup     51-61 ns    unforgeable, constant
#   ``any(t is k for k in ...)``   168-508 ns   unforgeable, by POSITION
#
# THE CHOICE IS BETWEEN THE TWO UNFORGEABLE SPELLINGS, and among those the
# index is the constant-time one. The `frozenset` is the cheapest and is
# the one that had to go, because ``t in <set>`` runs the METACLASS — that
# is S14, not a performance argument, and this comment should not be read
# as one.
#
# AND THE ``any()`` FIGURE IS A DISTRIBUTION, NOT A CONSTANT — audit 0.2.0
# B6 audit 8. This comment quoted ``487 ns`` for an `ir`-dataclass hit as
# though it were one number. It is the cost of the scan REACHING that
# type, so it is linear in where the hit lands in the set's iteration
# order — and that order comes from the type objects' addresses, which
# move between processes: measured, ``ir.Jaxpr`` sat at scan position 19
# of 20 in one process (508 ns) and at position 6 in the next (297 ns).
# ``487`` was a last-position reading quoted as a typical one, in a
# comment whose whole subject is that a scan cost depends on position.
#
# THE ROW SPLIT THAT COMMENT DREW DOES NOT REPRODUCE EITHER. It gave an
# exact-leaf hit and an `ir`-dataclass hit different `frozenset` costs
# (13.5 ns against 33.1 ns). Both are one hash lookup on a type object and
# they measure the same here within noise; what actually separates two
# readings of any row is scan position, which only the ``any()`` spelling
# has. The absolute numbers above are roughly twice that comment's on
# every row, so they are this machine's on this date and the RATIOS are
# what the argument rests on.
#
# ``_canonical`` is the hottest line in this module — every field of every
# `ir` object comes through it, and the largest traced query in this
# repo's own corpus builds **1204 `ir` objects from 112 equations**
# (measured 2026-08-17 over every zero-argument harness in
# `corpus/supply`; the objects-per-equation ratio across that corpus runs
# 5.36 to 10.83, and `CHANGELOG.md` names the harness — this file may not,
# under ARCHITECTURE.md Rule 2). That replaces *"a 141-equation traced
# query builds 868 `ir` objects"*, which named no query and matched none:
# nothing in this repo traces to 141 equations or to 868 objects, and a
# figure a reader cannot go and re-measure is the thing this batch keeps
# finding. A scan
# is linear in a set this module COMPUTES, so a dataclass added later
# lengthens every call; the dict is O(1) in that set and merges what were
# three lookups into one. What bounds the real cost is not any per-call
# figure here but the whole-document rows of the cost table in
# `CHANGELOG.md`, which are themselves a dated off-tree measurement.
_STORED_AS_IS: dict[int, type] = {}


def _rebuild_stored_index() -> None:
    """Recompute :data:`_STORED_AS_IS` from the three sets it merges.

    Called once at the foot of this module, where ``_CANONICAL_IR_TYPES``
    is computed, and again on every registration. It is the ONLY cache in
    this pass keyed on something other than a class object, so
    `tests/test_ir_canonicalization.py` re-derives it from the three sets
    rather than trusting that every mutator remembered to call this.
    """
    global _STORED_AS_IS
    _STORED_AS_IS = {
        id(t): t
        for t in (_CANONICAL_EXACT_TYPES
                  + tuple(_CANONICAL_IR_TYPES)
                  + _LIBRARY_STORED_TYPES)
    }


def _tuple_payload(v):
    """An exact `tuple` of what a `tuple` SUBCLASS actually carries."""
    return tuple.__getitem__(v, slice(None))


def _list_payload(v):
    """An exact `tuple` of what a `list` SUBCLASS actually carries.

    ``list.__getitem__`` builds an exact `list`, whose own ``__iter__``
    the ``tuple()`` then walks — so neither read is the subclass's.
    """
    return tuple(list.__getitem__(v, slice(None)))


def _read_or_refuse(read, obj: object) -> object:
    """A base type's own accessor applied to ``obj``, or ``obj`` refused.

    THE ONE THING A DERIVED CLASS CAN STILL DO TO ``issubclass``, and why
    the accessor is what settles it — audit 0.2.0 B6 audit 7.
    ``issubclass(type(obj), base)`` dispatches ``type.__subclasscheck__``
    on the BASE, so an override on the derived class cannot answer it; a
    metaclass CAN, however, override ``mro()`` and hand the interpreter a
    C-level MRO the class did not earn. Measured on this tree: CPython's
    own layout check refuses that for every base in
    :data:`_CANONICAL_READS` and for both of
    :data:`_SHAPE_PARAM_CONTAINERS` (*mro() returned base with unsuitable
    layout*), and permits it in exactly one direction — ``bool`` from a
    real ``int`` subclass, which shares ``int``'s layout and is therefore
    not a lie about the payload at all.

    That argument is about today's CPython, so it is not what this pass
    rests on. What it rests on is that an accessor handed an object
    without its type's payload RAISES, and this function turns that into
    the module's own refusal. The alternative is not a wrong value — it is
    ``descriptor '__getitem__' requires a 'tuple' object but received a
    'Liar'`` escaping a public constructor as a raw `TypeError`, which
    `TranscriptionError` SUBCLASSES, so ``except TranscriptionError`` does
    not catch it. Measured on `dff95fc` by driving a liar of each of the
    nine faces this door names, in each of the THREE SPELLINGS of the two
    bypasses — metaclass alone, ``__class__`` alone, and both together —
    into a plain param value and into both declaration params:
    ``9 x 3 x 3 = 81`` combinations, and the arithmetic is written out
    because *"each of the two bypasses"* over 81 combinations is a
    sentence that does not multiply (audit 0.2.0 B6 audit 8; the two
    mechanisms are two, the spellings driven are three, and a repair that
    closed only the conjunction is exactly what the third spelling exists
    to catch). That drive gave
    **19 raw `TypeError`s at SIX distinct statements**: the ``complex``
    and ``bytes`` reads, the shared ``read(obj)`` call, the ``tuple`` and
    ``list`` container arms, and :func:`_validate_decl_eqn`'s
    ``str.__str__`` on the ``dtype`` param. The message-totality sweep
    cannot reach any of them, because every leaf it injects is a real
    SUBCLASS and none of these objects is a subclass of anything. On
    `main` the same 81 combinations produce none — that door does not
    exist there — so this half is a regression of `dff95fc` and not a
    defect of the release.
    """
    try:
        return read(obj)
    except Exception:  # noqa: BLE001 — an unreadable payload IS the finding
        raise _NotCanonical(obj) from None


def _canonical(obj: object) -> object:
    """``obj`` as an EXACT instance of a type this module stores, or as
    one of the two kinds of object it CARRIES with the delegation stated
    (one of this module's own frozen dataclasses; a registered library
    type).

    Raises :class:`_NotCanonical` — never a `TranscriptionError` — because
    the path back to the field is composed as the signal unwinds and only
    the caller that owns a `where` can turn it into a refusal.

    **EVERY DECISION HERE IS MADE ON ``type(obj)``, BY IDENTITY OR BY
    ``issubclass`` AGAINST THE BASE, AND NOTHING ELSE WILL DO** — audit
    0.2.0 B6 audit 7. The contract line above was true of what this
    function DID with a value and false about which values it did it to:
    the dispatch was ``type(obj) in <a frozenset>``, which runs the
    METACLASS, and ``isinstance(obj, base)``, which reads the object's
    own ``__class__``, so an object could be stored having reached none
    of the mechanism described here. See the door's narrative above.
    """
    t = type(obj)
    if id(t) in _STORED_AS_IS:
        # STORED AS IS, and the three reasons are one lookup because they
        # are one answer: an exact leaf built-in is already what it will
        # be stored as; one of this module's own frozen dataclasses is
        # canonical by its OWN `__post_init__`, which ran before this
        # object could be handed to anything, so there is nothing to
        # recurse into and a document costs one pass over its own size
        # rather than one per level of nesting; and a registered library
        # type is carried by the declaration another `stelling` module
        # made. The registered arm is settled HERE, before the subclass
        # reads below, because registration says "carry this" and a
        # registered type that happened to subclass a stored one would
        # otherwise be rewritten into something its own module did not
        # build — and `_register_stored_type` refuses a type this index
        # already holds, so merging the three cannot move which arm
        # decides anything.
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
    for base, read in _CANONICAL_READS:
        if issubclass(t, base):
            return _read_or_refuse(read, obj)
    # the recursion is OUTSIDE the guarded read on purpose: a
    # `_NotCanonical` raised by an ELEMENT is a different finding about a
    # different object, and must keep its own path back to the field
    if issubclass(t, tuple):
        return _canonical_items(_read_or_refuse(_tuple_payload, obj))
    if issubclass(t, list):
        return _canonical_items(_read_or_refuse(_list_payload, obj))
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

    For the places that need the CONTAINER settled before its contents
    are: the ``params`` sequence and each ``(key, value)`` entry in it,
    whose keys must be settled before the sort and whose values must not
    be, because :func:`_validate_decl_eqn` owns two of them — and, since
    audit 0.2.0 B12, EVERY sequence position of a document, through
    :func:`_doc_sequence`, which is this function plus the position it is
    being read at. :data:`_DOC_SEQUENCE_CONTAINERS` names the pair this
    accepts so a refusal can quote it; `tests/test_document_schema.py`
    computes the pair from this function rather than trusting that name.
    """
    t = type(v)
    if t is tuple:
        return v
    if t is list:
        return tuple(v)
    # `issubclass` against the BASE, never `isinstance` against the
    # object, for the reason :func:`_canonical` gives — this function has
    # the same two arms and had the same two holes (audit 0.2.0 B6 audit 7)
    if issubclass(t, tuple):
        return _read_or_refuse(_tuple_payload, v)
    if issubclass(t, list):
        return _read_or_refuse(_list_payload, v)
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


# --- the field's own declared type ------------------------------------------
#
# THE SECOND HALF OF THE ONE RULE, AND THE HALF THAT NEEDED NOTHING WRITTEN
# — audit 0.2.0 B12. The canonicalization door above asks *"is there an
# exact form to store this value as"*, which is a question about
# SINGLE-VALUEDNESS and is answered YES for every exact built-in. It is not
# the question *"is this the type this position holds"*, and nothing asked
# that one: `Var(id=None)`, `Aval(weak_type="xx")`, `Aval(dtype=1.5)`,
# `DebugInfo(func=0)` and — the one that mints a false VERIFIED —
# `JaxprEqn(primitive=None)` were all stored, and round-tripped faithfully,
# on `main` at `a4e4056`.
#
# THE DECLARED TYPE IS THE FIELD'S ANNOTATION, WHICH IS ALREADY IN THIS
# FILE. `primitive: str`, `id: int`, `weak_type: bool`, `dtype: str | None`,
# `eqns: tuple[JaxprEqn, ...]`, `fields: tuple[tuple[str, object], ...]` —
# every one of them a statement nobody has to write twice, because
# :func:`typing.get_type_hints` reads them off the class. A field added to
# one of these dataclasses later is judged by its own annotation with
# nothing added here, which is the same property :func:`_canonicalise`
# already has and the reason neither is a table.
#
# WHY THE TEST IS EXACT (``type(v) is t``) AND NOT ``isinstance``. Every
# value that reaches it has been through :func:`_canonical` on the line
# before, so it is an exact built-in, an exact tuple, one of this module's
# own frozen dataclasses, or a registered library type — an `isinstance`
# would therefore admit exactly the same values while reading the OBJECT's
# ``__class__``, which is one of the two bypasses audit 7 measured, so it
# would be strictly worse for nothing. And it would be wrong at one
# position for a reason that has nothing to do with liars: `bool` IS an
# `int` subclass, so ``isinstance(True, int)`` is True and `<var>.id: true`
# — a document no encoder writes — would pass. (The mirror case does not
# arise: ``isinstance(1, bool)`` is already False, so `<aval>.weak_type: 1`
# is refused either way.)
#
# AND THE MEMBERSHIP IS BY IDENTITY, never ``in`` and never ``==``, for the
# reason :data:`_STORED_AS_IS` gives: ``type(v) in <a tuple of types>`` runs
# the METACLASS, which is S14.
#
# WHAT THE ANNOTATION CANNOT SAY, and it is one thing: the registered
# library type a `ClosedJaxpr.consts` entry may hold in place of a value
# (`interval.IntervalArray` — see :data:`_LIBRARY_STORED_TYPES`). This
# module may not import the module that defines it, so `consts`' annotation
# cannot name it and does not. A registered type is therefore accepted at
# ANY field, uniformly, because registration IS the declaration that this
# module carries the value without judging it — and because no document can
# reach one: `_decode` has NO TAG for it, so no JSON document produces one
# and the only way one arrives is a caller who built it and put it there.
# Scoping the exception to `consts` would be the enumeration this whole
# door replaces, and would be a narrower claim than the truth.
#
# THE ARGUMENT RESTS ON `_decode` ALONE, and it used to rest on `_encode`
# as well — *"`_encode` refuses to encode one"* — which is false at the
# positions this exception is widest at. B12's own review, on B12's tree.
# `_encode` refuses a registered value only in the arms where it RECURSES:
# a `ClosedJaxpr` whose ``consts`` hold one raises ``stelling.ir cannot
# encode IntervalArray``, correctly. At a slot it writes STRAIGHT THROUGH
# with no type test, ``to_dict()`` does NOT raise — it returns a dict with
# the `IntervalArray` sitting in the slot. **EIGHTEEN POSITIONS, MEASURED
# BY DRIVING EVERY ONE OF THEM**, because an enumeration written by hand
# was wrong the first time this paragraph tried it:
#
#     <aval>.kind  <aval>.dtype  <aval>.weak_type  <var>.id
#     <enum>.cls  <enum>.member  <sentinel>.cls  <opaque>.cls
#     <treedef>.text  <ntuple>.cls  <ntuple>.fields KEY
#     <eqn>.primitive  <eqn>.effects[*]  <eqn>.source_info[*]
#     <dbg>.func  <dbg>.arg_names[*]  <dbg>.result_paths[*]
#     <jaxpr>.effects[*]
#
# The non-recursing slots NOT in that list never get that far, because a
# registered value at them is refused AT CONSTRUCTION: `<aval>.shape[*]`
# and `<array>.shape[*]` by the extent rule (`_load_extents` wants
# ``__index__``), `<eqn>.params` KEYS by `_canonical_param_keys` — all
# three a `TranscriptionError` — and `<array>.dtype` / `<array>.data` by
# `Array.__post_init__`'s own ``len()`` reads, which raise a plain
# `TypeError`. `<complex>.re`/`.im` are read off a `complex` object, so
# nothing else can be there at all.
#
# (`content_hash` raises on such an object at FOURTEEN of the eighteen —
# from `json.dumps`, which is loud and is not this module deciding
# anything — and ANSWERS at the four the hash does not cover:
# `<eqn>.source_info[*]` and the three `<dbg>` slots are exactly what
# ``to_dict(include_metadata=False)`` omits, and `content_hash` is defined
# over that reduced document. A correct hash of a scope that excludes
# them, not a refusal; this line said *"does still raise"* unqualified
# until B12's own re-review.) The conclusion
# is unharmed because it never needed the `_encode` half: a document is
# JSON, JSON reaches this module through `_decode`, and `_decode` has no
# tag. `tests/test_document_schema.py` pins that half, drives all
# eighteen positions above, and checks its own position set against
# `_encode`'s AST — so neither the argument nor the enumeration is a
# sentence anybody has to re-derive. **THAT AST CHECK COMPARES THE FULL
# ``<tag>.key``, and it used to compare the bare key name**: a slot added
# to the encoding under a key name already driven at another tag was
# invisible to it, measured at `e3fe0fb` — a new ``<aval>.cls`` passed
# while a new ``<aval>.zzz`` failed. It also read a dict value as a
# recursion whenever the text ``_encode(`` appeared anywhere in it, which
# dropped the two positions where the VALUE recurses and the KEY does not
# (`<eqn>.params`, `<ntuple>.fields`) — so those two stood in the hand-
# written list on BOTH sides of the comparison, driven but never derived,
# which is the one thing the check exists to prevent. Both corrected in
# B12's own re-review; the classification is structural now, and
# conservative, so an encoding shape it does not recognise lands in the
# population rather than out of it.
#
# THE FIELDS WITH A STRONGER RULE OF THEIR OWN ARE SKIPPED, exactly as they
# already were: an `Aval`'s and an `Array`'s ``shape`` (`_load_extents`
# reads each extent through ``__index__`` and installs a plain `int`, which
# is strictly more than ``tuple[int, ...]`` asks) and a `JaxprEqn`'s
# ``params`` (`_canonical_param_keys` owns the entries and the keys,
# `_validate_decl_eqn` owns two of the values). Each of those rules
# nonetheless PRODUCES what the annotation states, and that is measured
# rather than asserted: `tests/test_document_schema.py` walks a real traced
# query and checks every field of every object against its own spec, the
# skipped ones included.
_FIELD_SPECS: dict[type, dict[str, tuple]] = {}


def _spec_of(ann: object) -> tuple:
    """A field annotation as the shape :func:`_matches_spec` reads.

    Four shapes, which is every shape the annotations in this module use:
    ``("any",)`` for ``object``; ``("exact", (types…))`` for a type or a
    union of types; ``("seq", spec)`` for ``tuple[X, ...]``; and
    ``("pair", (spec, …))`` for a fixed-length ``tuple[A, B]``. An
    annotation of any other shape raises, loudly and at first construction,
    rather than being silently treated as unconstrained — a field this
    function cannot read is a field nothing would be checking.
    """
    if ann is object:
        return ("any",)
    origin = _typing.get_origin(ann)
    if origin is tuple:
        args = _typing.get_args(ann)
        if len(args) == 2 and args[1] is Ellipsis:
            return ("seq", _spec_of(args[0]))
        return ("pair", tuple(_spec_of(a) for a in args))
    if origin is _types.UnionType or origin is _typing.Union:
        members: list[type] = []
        for a in _typing.get_args(ann):
            sub = _spec_of(a)
            if sub[0] != "exact":
                raise TypeError(
                    f"stelling.ir: the field annotation {ann!r} is a union "
                    f"with a non-type member {a!r}; _spec_of reads unions of "
                    f"types only"
                )
            members.extend(sub[1])
        return ("exact", tuple(members))
    if isinstance(ann, type):
        return ("exact", (ann,))
    raise TypeError(
        f"stelling.ir: the field annotation {ann!r} is of a shape _spec_of "
        f"does not read, so nothing would be checking that field"
    )


def _spec_rule(spec: tuple) -> str:
    """A spec as a sentence, DERIVED from the spec — so a refusal can never
    name a different type from the one it applied. The discipline
    :data:`_CANONICAL_RULE` and :data:`_SHAPE_PARAM_RULE` carry."""
    kind = spec[0]
    if kind == "any":
        return "any stored value"
    if kind == "exact":
        return " or ".join(
            "None" if t is type(None) else t.__name__ for t in spec[1]
        )
    if kind == "seq":
        return f"a tuple of ({_spec_rule(spec[1])})"
    return "a tuple of (" + ", ".join(_spec_rule(s) for s in spec[1]) + ")"


def _matches_spec(spec: tuple, v: object) -> bool:
    """Is ``v`` of the type :func:`_spec_of` read off the field?

    Membership is ``type(v) is t`` — identity against the real type object,
    which no metaclass and no ``__class__`` property takes part in (S14).
    The only value accepted against a spec it does not match is a
    :data:`_LIBRARY_STORED_TYPES` registration, for the reason the
    narrative above gives, and that arm is reached only after the
    annotation has already said no.
    """
    kind = spec[0]
    if kind == "any":
        return True
    t = type(v)
    if kind == "exact":
        if any(t is s for s in spec[1]):
            return True
    elif t is tuple:
        if kind == "seq":
            inner = spec[1]
            return all(_matches_spec(inner, x) for x in v)
        return len(v) == len(spec[1]) and all(
            _matches_spec(s, x) for s, x in zip(spec[1], v)
        )
    return any(t is r for r in _LIBRARY_STORED_TYPES)


def _canonicalise(obj, where: str, *, skip: tuple[str, ...] = ()) -> None:
    """Replace every field of an IR dataclass with its exact-typed twin,
    and refuse a field whose value is not of the type the field DECLARES.

    The field names come from :func:`dataclasses.fields` and their declared
    types from :func:`typing.get_type_hints`, both cached per class: a field
    added to one of these dataclasses later is canonicalized AND judged
    without anyone having to add it here, which is the difference between
    this and a repair per member (audit 0.2.0 B12 for the second half; see
    the narrative above it).

    The hints are resolved LAZILY, at first construction of each class,
    and that is a choice about where the code lives rather than a
    necessity: every annotation in this module names something defined
    ABOVE it, so a pass at the foot of the module beside
    :data:`_CANONICAL_IR_TYPES` would resolve them too. Doing it here keeps
    the field list and the spec table in one place and costs one dict
    lookup per construction; it is safe because nothing in this module
    constructs IR at import, so the namespace is complete by the time any
    of this runs.
    """
    cls = type(obj)
    names = _field_names(cls)
    specs = _FIELD_SPECS.get(cls)
    if specs is None:
        hints = _typing.get_type_hints(cls)
        specs = _FIELD_SPECS[cls] = {n: _spec_of(hints[n]) for n in names}
    for name in names:
        if name in skip:
            continue
        v = getattr(obj, name)
        try:
            c = _canonical(v)
        except _NotCanonical as exc:
            exc.path.append(f".{name}")
            _refuse_uncanonical(exc, where)
        spec = specs[name]
        if not _matches_spec(spec, c):
            # COMPOSED ONLY ON THE FAILING PATH, for the reason
            # `_canonical_param_keys` gives: a `_load_check` message is an
            # ARGUMENT, `repr()` of a document-supplied value is a READ of
            # it, and the passing path of this door is exactly one read per
            # value.
            _load_check(
                False,
                f"{where}.{name}",
                f"value {_safe_repr(c)} is of type {_safe_type_name(c)}, and "
                f"this field is declared {_spec_rule(spec)}: that "
                f"declaration is what every reader of this IR relies on, and "
                f"this position is document-supplied, so a value of any "
                f"other type is not a fact about the program the document "
                f"describes — it is a different program, silently "
                f"substituted",
            )
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
    # `issubclass(type(v), C)` AND NOT `isinstance(v, C)` — audit 0.2.0 B6
    # audit 7, and the reason is the same at every site in this pass that
    # spells it this way. `isinstance` falls back to reading the OBJECT's
    # `__class__`, which a two-line property answers, and each of the two
    # branches below GATES a check: a value that claims to be neither an
    # `Array` nor a scalar has its shape claim cross-checked against
    # nothing at all. `issubclass` dispatches on the BASE, whose metaclass
    # is `type`, so the derived class does not get a say. The door in
    # front of this function refuses such an object already — and this
    # function is annotated above as not resting on another guard having
    # run, so it may not rest on that one either.
    if issubclass(type(val), Array):
        val_dims = _validate_array_value(val, where)
        _load_check(
            val_dims == aval_dims,
            where,
            f"array value shape {val_dims} contradicts the "
            f"recorded aval shape {aval_dims}",
        )
    elif issubclass(type(val), (bool, int, float, complex)):
        _load_check(
            aval_dims == (),
            where,
            # a document-supplied scalar, and a TYPE test is a claim about
            # the type: an `int` subclass whose `__repr__` raises
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
    # a TYPE test and not an `isinstance` — see
    # :func:`_validate_value_against_aval`; an atom that claims to be
    # neither has its aval validated by nothing
    if issubclass(type(atom), Var):
        _validate_aval(atom.aval, where)
    elif issubclass(type(atom), Literal):
        _validate_aval(atom.aval, where)
        _validate_value_against_aval(atom.val, atom.aval, where)


def _validate_param_value(v, where: str) -> None:
    # a TYPE test and not an `isinstance` — see
    # :func:`_validate_value_against_aval`. Here the branch decides
    # whether to RECURSE, so a param that claims to be none of these
    # carries a whole nested jaxpr past the walk
    if issubclass(type(v), ClosedJaxpr):
        _validate_closed(v, where)
    elif issubclass(type(v), Jaxpr):
        _validate_jaxpr(v, where)
    elif issubclass(type(v), Array):
        _validate_array_value(v, where)
    elif issubclass(type(v), (tuple, list)):
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
    elif issubclass(type(v), NamedTupleParam):
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
        _validate_decl_nonempty(eqn, w)


# The param keys jax supplies on EVERY equation for these primitives, measured
# by tracing 42 distinct forms on jax 0.11.0 and checking the key set is
# constant per primitive (`param_key_census.py` in the sweeps repository; the
# census reports which primitives it reached, and a primitive absent from this
# table is simply unconstrained here).
#
# RE-DRIVEN ON THE OTHER TESTED SERIES 2026-08-07, because a required-key
# table measured on one series is a load-path refusal aimed at the other.
# 21 forms on jax 0.10.2 reached 18 of the 19 entries this table then had
# (all but `stelling_any`, which is stelling's own and never jax-traced):
# every required key present, and the key set constant per primitive,
# exactly as on 0.11.0. `scan` is the standing counterexample to assuming
# a result travels: its key set is disjoint across those two series
# (`num_consts`/`num_carry` vs `ft_in`/`ft_out`), and it is not in this
# table.
#
# "THIS TABLE IS SERIES-STABLE SO FAR" — WITHDRAWN, 2026-08-18, BY A
# COUNTEREXAMPLE INSIDE A SERIES. That sentence stood here and told a reader
# to "re-drive it per series", and jax 0.11.1 falsified the unit: 70 forms
# re-driven on 0.11.0 and on 0.11.1 (this machine, CPython 3.12.3, x64
# enabled) each reached 60 distinct primitives and all 20 entries of the
# table below, with every required key present and every primitive's key set
# constant across the 70 forms on both — and with EXACTLY TWO primitives
# disagreeing between the two releases:
#
#     reduce_max   0.11.0: {axes}   0.11.1: {axes, out_sharding}
#     reduce_min   0.11.0: {axes}   0.11.1: {axes, out_sharding}
#
# 0.11.0 and 0.11.1 are one series. So: RE-DRIVE THIS TABLE PER RELEASE, not
# per series. A series is not the unit at which jax's param sets hold still,
# and this comment claimed it was.
#
# WHY THE TWO MOVERS GET NO ROW, which is the load-bearing half. A row here
# is a REFUSAL AIMED AT A STORED DOCUMENT, and a document is loaded by a
# different jax from the one that traced it — that is the whole reason
# documents exist. `reduce_max: {axes, out_sharding}` would therefore refuse
# an honest `jnp.max` query traced on jax 0.11.0 — measured, on a document
# this repository's own `to_dict` wrote, refused on both releases with the
# row present and loading on both without it. A key may only be required
# once it is present on every release a document could have come from, and
# `out_sharding` on these two arrived in 0.11.1: how far back upstream it
# exists was not measured, only that 0.11.0 does not emit it.
# `reduce_sum` may carry it because the 2026-08-07 re-drive
# above found every required key of every row it reached present on 0.10.2,
# it was measured on 0.11.0 when the row was written, and the re-drive
# above measures it present on 0.11.1. Absence from this table means
# "unconstrained", which is the true statement for a key that some supported
# releases supply and others do not.
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


# binary64's own infinity. Exact, and compared by ``==`` rather than by
# ``is`` — two `float("inf")`s are equal and need not be one object.
_INFINITY = float("inf")


def _validate_decl_bounds(params: dict, where: str) -> dict[str, object]:
    """A ``stelling_any`` declaration's ``lo``/``hi``: the TYPE, and the
    exact ``float`` each was read as, RETURNED to be installed.

    EVERY CONSTRUCTION PATH, exactly like the ``shape`` and ``dtype`` rules
    beside it and for the same reason — the install is what makes the value
    a later reader gets the value this function judged, and a rule that ran
    only on the load path could not install anything.

    Split out of :func:`_validate_decl_eqn` rather than inlined because it
    is the only part of that function that does not depend on the outvar
    aval, and the difference is load-bearing — see the comment at the call.

    THE SET being non-empty is a different question with a different scope
    and lives in :func:`_validate_decl_nonempty`, on the load path only.
    """
    out: dict[str, object] = {}
    for name in ("lo", "hi"):
        if name not in params:
            continue
        raw = params[name]
        # CANONICALIZED FIRST, then a TYPE test on what the door produced —
        # exactly as the `dtype` param is, and for the same reason: this
        # function runs BEFORE `_canonical_param_values`, so `raw` is still
        # the object the document handed over, and an `isinstance` here
        # would take that object's word for it.
        try:
            v = _canonical(raw)
        except _NotCanonical as exc:
            exc.path.append(f".params[{name!r}]")
            _refuse_uncanonical(exc, where)
        if type(v) is not float:
            # composed only on the failing path (see `_canonical_param_keys`)
            _load_check(
                False,
                where,
                f"stelling_any {name} param {_safe_repr(raw)} is of type "
                f"{_safe_type_name(raw)}, and a declared bound is recorded as "
                f"binary64: `harness.any_array` stores the binary64 image of "
                f"whatever spelling the caller wrote, so `float` is the type "
                f"`to_dict` writes here — and every reader of this param "
                f"(`propagate`'s four declaration transfers, "
                f"`obligation`'s slicer, `vacuity.widen`) either calls "
                f"`float()` on it or compares it raw, so a param of any other "
                f"type is not a bound this function can judge, it is two "
                f"readers about to disagree",
            )
        out[name] = v
    return out


def _validate_decl_nonempty(eqn: "JaxprEqn", where: str) -> None:
    """A LOADED ``stelling_any`` must declare a NON-EMPTY set.

    THE REFUSAL :func:`stelling._jax_compat.any_array` ALREADY MAKES, made
    at the deserialization door too — audit 0.2.0 B12, S16. An empty
    declared set verifies every universal claim over it vacuously, which is
    audit-gate finding 3 and the reason the trace face refuses it; measured
    on `main` at `a4e4056`, a persisted query whose declaration was edited
    to `lo:inf, hi:inf` returned **VERIFIED with 100% coverage** where the
    honest verdict is UNKNOWN.

    ``lo <= hi`` COVERS NaN BY CONSTRUCTION, which is stated because it
    looks like an omission: ``nan <= x`` is False for every ``x``, so a NaN
    endpoint fails the same comparison an inverted pair fails, and both are
    the same fact — the declaration describes no value at all. `any_array`
    spells it ``not lo <= hi`` for the same reason and says so.

    **LOAD PATH ONLY, DELIBERATELY, AND NOT FOR THE REASON
    :func:`_validate_required_params` GIVES.** That one is load-only because
    hand-built IR legitimately omits params. This one is load-only because
    the two faces are asking about different things. `any_array` is the
    DECLARATION API — the sentence a user writes — and a document claims to
    be a persisted traced query, so it may not carry a declaration that API
    would have refused. `ir.JaxprEqn` is the constructor underneath both,
    and the suite builds infinite and NaN declarations through it on
    purpose, over operands no `any_array` will produce. Refusing those at
    construction would delete a tested capability to close a document
    surface, which is a wider change than the defect asks for.

    **HOW MUCH CAPABILITY, MEASURED RATHER THAN NAMED**, because the first
    spelling of this paragraph named one file and the count is four.
    Calling this function from `JaxprEqn.__post_init__` instead — the
    whole of the change — turns **11 pre-existing tests red across four
    files**: 7 in `tests/test_ieee_semantics.py`, whose ``(inf, inf)``
    declarations drive the maybe-NaN flag through `ne`/`eq`/`pow`/`min`/
    `max`; 2 in `tests/test_transfers.py`, the non-finite
    `scatter`/`gather` index declines; 1 in
    `tests/test_ieee_zero_divisor_and_mul_exact.py`; and 1 in
    `tests/test_undecided_detail.py::test_declined_declaration_side_points_at_the_declaration`,
    **which is where the ``(nan, hi)`` declaration actually lives**
    (``any_eqn(x, nan, 4.0)``), and not in the ieee file this paragraph
    used to credit it to. That citation is on ONE line, past 79 columns,
    because `tests/test_prose_hygiene.py` reads a `path::name` reference a
    line at a time: wrapped mid-token, as this one was until B12's own
    re-review, the pairing the sentence asserts is pinned by nothing.
    The capability itself is pinned in one place —
    `tests/test_document_schema.py::test_the_emptiness_rule_is_the_LOAD_doors_and_not_the_constructors`
    constructs a witness for each of the two refusals in each direction —
    so the argument rests on a test in this repository rather than on a
    list of other files' names.

    **AND THAT MAKES THOSE OBJECTS CONSTRUCTIBLE AND NOT RELOADABLE**,
    which is the price of the choice and is stated HERE because this is
    where a reader will look for it. ``ClosedJaxpr.from_dict(cj.to_dict())``
    raises `TranscriptionError` for every declaration this function
    refuses: `to_dict` encodes it without complaint, and the document it
    writes is one `from_dict` will not take back. The same has always been
    true of :func:`_validate_required_params`' subject — a params-less
    equation — and those two are the whole of the exception, enumerated
    from this file's call graph by `tests/test_document_schema.py` rather
    than from this sentence — over the WHOLE load path, `_decode` and
    :func:`_validate_loaded` both, which is a correction B12 made to its
    own instrument after seeding it from the validator alone. It is not a
    soundness hole: the refusal is loud, mints no verdict, and leaves
    `content_hash` — a function of `to_dict` alone — untouched. It is a
    bound on what "round-trips" means, the serialization comment above
    :func:`_encode` claimed there was none, and the batch that added this
    rule widened the bound by two classes without noticing. Both are
    corrected there.

    It also leaves something unsettled that this function is not the place
    to settle, recorded so the next reader does not mistake it for an
    oversight: `any_array`'s ``(inf, inf)`` refusal is argued from the
    stamped ℝ semantics, under which an infinite endpoint means unbounded —
    and stelling has a registered ``semantics="ieee"`` dial under which
    ``[inf, inf]`` is the perfectly ordinary point ``{+inf}``. The dial is
    chosen at check time and is not recorded in the IR, so neither face can
    read it. This function matches the trace face because a document is a
    persisted product of that face; whether that face should itself be
    dial-aware is a question about `_jax_compat.any_array` and is reported
    rather than answered here.

    WHAT THE TRACE FACE ALSO REFUSES AND THIS ONE CANNOT: a bound that
    binary64 cannot hold without NARROWING the declared set. That judgment
    needs the dtype's own value grid and the caller's exact value, and
    neither exists here — a document carries the recorded binary64 and
    nothing else, so the narrowing has already happened or has not. That is
    a bound on what this function claims, not a hole it leaves open: the
    stored endpoints ARE the set the analyses reason about, and the two
    refusals below are the whole of what makes that set non-empty.
    """
    if eqn.primitive != "stelling_any":
        return
    params = dict(eqn.params)
    if "lo" not in params or "hi" not in params:
        return
    lo, hi = params["lo"], params["hi"]
    # BOTH MESSAGES ARE COMPOSED ONLY ON THE FAILING PATH, for the reason
    # `_canonical_param_keys` gives: a `_load_check` message is an ARGUMENT
    # and `repr()` of a stored value is a READ of it, so composing one on
    # every load of every declaration would put two reads on a path whose
    # whole property is that it has one. The quotes are guarded even so —
    # both endpoints are exact `float` by `_validate_decl_bounds`, which ran
    # in this equation's own `__post_init__`, but this pass does not rest on
    # another guard having run and that includes its own.
    if not lo <= hi:
        _load_check(
            False,
            where,
            f"stelling_any bounds ({_safe_repr(lo)}, {_safe_repr(hi)}) "
            f"declare an EMPTY set, which verifies every universal claim "
            f"over it vacuously and refutes every existential one — the "
            f"refusal `harness.any_array` makes at declaration time, made "
            f"here because a document claims to be a query that came "
            f"through it",
        )
    if lo == hi and (lo == _INFINITY or lo == -_INFINITY):
        _load_check(
            False,
            where,
            f"stelling_any bounds ({_safe_repr(lo)}, {_safe_repr(hi)}) "
            f"declare an empty REAL set: under the stamped ℝ semantics "
            f"an infinite endpoint means unbounded, so an infinite point "
            f"has no members (the same refusal `harness.any_array` makes)",
        )


def _validate_decl_eqn(eqn: "JaxprEqn", where: str) -> dict[str, object]:
    """A stelling_any declaration's aval must agree with its OWN
    params — two self-descriptions of one declared set. Called from
    JaxprEqn.__post_init__ (every construction path) and from the
    from_dict walk (kept for its load-path error context).

    **RETURNS THE PARAMS IT VALIDATED, NORMALISED, KEYED BY NAME** — the
    ``shape`` param's extents as plain ``int`` in a plain ``tuple``, the
    ``dtype`` param as an exact ``str``, and ``lo``/``hi`` as exact
    ``float`` — and an EMPTY dict when the equation carries none of them to
    validate (a form this function deliberately still blesses; see the last
    paragraph).
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
    :meth:`stelling.obligation._Slicer._declared_shape` makes. A param of
    the wrong container type is refused for its TYPE; a param of the right
    type whose iteration raises (a ``list`` SUBCLASS whose ``__iter__``
    refuses — ``isinstance`` is a claim about the type and not about the
    object) is refused for being UNREADABLE. One sentence for both said
    "is not a sequence of extents" over ``np.array([4])``, which is a
    sequence of extents and was refused for a different reason.

    **``lo`` AND ``hi`` ARE THE DECLARED SET ITSELF, AND THEY HAD NO RULE AT
    ALL** — audit 0.2.0 B12, S16. Three things were missing and they are one
    finding: a TYPE (the IR records a bound as binary64, and `"0.5"`,
    `true`, `1` and `null` were all admitted, at ten reader sites that each
    do ``float(params["lo"])`` with no gate in front of them, of which
    ``""``, ``"xx"``, ``null`` and ``()`` each reached a RAW crash out of
    the public `propagate()`); the INSTALL
    (the guard read one object and every reader re-read another — and two of
    those readers DISAGREED about one value, `vacuity.widen` deciding *"is
    this a POINT declaration?"* with ``params["lo"] != params["hi"]`` on the
    raw objects while every analysis reads ``float(...)``, so `lo:"1.0"` /
    `hi:1.0` is a point by the reading every analysis makes and not a point
    by the reading `widen` makes, and was widened by the mode whose whole
    contract is that a point declaration is held still); and the EMPTINESS
    refusal
    :func:`stelling._jax_compat.any_array` already makes on the trace face,
    which is :func:`_validate_decl_nonempty` and is load-path only for the
    reason written there.

    The type is ``float`` and not "a number", because this face reads the
    ENCODING and not a caller's spelling: `any_array` records
    ``binary64_image(...)``, which is a python `float` for every accepted
    spelling, so `float` is exactly what `to_dict` writes at these two
    positions. The wide spelling family lives at the trace face, where a
    caller writes python — see :mod:`stelling._bound_spelling`, whose whole
    subject is that judging a bound by its TYPE while its VALUE goes through
    ``float()`` is how this layer's three earlier escapes happened. Here
    there is no exact value to preserve: the document's `float` IS the
    binary64 the parent recorded, so the two clauses `any_array` needs for a
    finite value whose image is infinite have nothing to say at this face
    and are not restated.

    THAT NARROWING IS CONFINED TO HAND-BUILT IR, exactly as the ``shape``
    and ``dtype`` rules above are, and there it is loud: no traced query and
    no `to_dict` output carries anything but a `float` here — measured over
    the whole accepted spelling family (python `int`/`bool`/`float`, numpy
    integer/floating/`longdouble`, `Decimal`, `Fraction`, the signed zeros
    and the infinities), every one of which `any_array` records as a python
    `float`. What it costs
    is that a hand-written `JaxprEqn` must spell an integer bound `4.0`
    rather than `4`, which is what `any_array` would have recorded for it
    anyway.

    **This door is not the soundness boundary and closing it does not make
    one.** `ir.py` scopes per-primitive shape inference out of the load
    validation in writing, and `stelling.obligation._Slicer.
    _one_shape_per_value` is what actually stands between a lying document
    and a verdict — including for the form this function deliberately
    still blesses, a declaration with NO ``shape`` param at all (hand-built
    IR legitimately omits params; see `_validate_required_params`). What
    this closes is the constructor's own claim to have compared the two
    self-descriptions when it had not."""
    if eqn.primitive != "stelling_any":
        return {}
    params = dict(eqn.params)
    normalised: dict[str, object] = {}
    # THE BOUNDS FIRST, AND WITHOUT AN OUTVAR IN THE CONDITION. `shape` and
    # `dtype` are the declaration's second self-description OF ITS OUTVAR
    # AVAL, so there is nothing to compare them against when there is no
    # outvar; `lo`/`hi` are the declared SET, which an equation carries
    # whether or not anything is bound to it. The guard used to read
    # ``primitive != "stelling_any" or not eqn.outvars``, which would have
    # left these two unjudged on an outvar-less declaration — a form
    # `_validate_required_params` does not forbid.
    normalised.update(_validate_decl_bounds(params, where))
    if not eqn.outvars:
        return normalised
    # (the SET those two bounds describe is judged on the load path, by
    # `_validate_decl_nonempty` — see that function for why the two
    # questions have different scopes)
    # the declaration's aval must agree with its OWN params —
    # the P1 arc's lies all started at a declaration whose two
    # self-descriptions disagreed
    if "shape" in params:
        shape = params["shape"]
        # THE RULE, ASKED OF THE ONE OBJECT THAT HOLDS IT, THROUGH THE ONE
        # FUNCTION THAT READS IT. The slicer's `_declared_shape` calls
        # `ir._held_in_a_shape_param_container` too, so the two faces
        # cannot part company by one of them being edited — nor by one of
        # them being hardened, which is how they parted last time (audit
        # 0.2.0 B6 audit 7; see that function).
        held_in_a_container = _held_in_a_shape_param_container(shape)
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
    if "dtype" in params and eqn.outvars[0].aval.dtype is not None:
        raw_dtype = params["dtype"]
        # THE PARAM'S TYPE IS PART OF THE AGREEMENT, AND IT WAS THE GATE ON
        # IT — audit 0.2.0 B6 audit 7. This read
        # ``if isinstance(raw_dtype, str)``, so a `dtype` param that is any
        # other exact built-in — the door stores `bytes`, `int`, `float`
        # and `tuple` perfectly happily — did not merely fail the
        # comparison, it SKIPPED it: measured, ``b'float64'``, ``0`` and
        # ``('float64',)`` were all ACCEPTED under a ``float64`` aval while
        # a ``str`` ``'int64'`` was correctly refused. So *"a declaration's
        # two self-descriptions must agree"* held only when the param
        # happened to be a `str`, which is a condition no document has to
        # meet.
        #
        # AND A FOURTH SPELLING, WHICH THIS GUARD'S OWN CONDITION MOVED —
        # audit 0.2.0 B6 audit 8. The line above was
        # ``if params.get("dtype") is not None``, so `.get` was using
        # ``None`` as its OWN sentinel and could not tell a `dtype` param
        # that is ABSENT from one that is present and ``null``. Changing it
        # to ``"dtype" in params`` is therefore a BRANCH-SELECTION change
        # and not only a type check, and the enumeration in the paragraph
        # above — ``b'float64'``, ``0``, ``('float64',)`` — does not
        # contain the document it moved. Measured at ``JAX_ENABLE_X64=1``,
        # on a document whose declaration carries ``["dtype", null]``, from
        # this harness — round-tripped through ``to_dict()`` with that one
        # param set to ``null``:
        #
        #     def h():
        #         x = any_array((2,), "float64", (0.0, 1.0))
        #         return assert_(x >= 0.0)
        #
        # TWO DOCUMENTS COME OUT OF THAT HARNESS AND THIS BLOCK IS ABOUT
        # BOTH, WHICH IT DID NOT SAY UNTIL 2026-08-23. Name them:
        #
        #     the TRACED document   what tracing the harness produces
        #     the NULLED document   that document round-tripped through
        #                           ``to_dict()`` with the declaration's
        #                           ``dtype`` param set to ``null``
        #
        # The ACCEPT/REFUSE property this whole block exists for is about
        # the NULLED one — it is the document ``_validate_decl_eqn``'s
        # branch-SELECTION change moved. Every ``content_hash`` figure
        # below that is a figure of THIS TREE'S is about the TRACED one,
        # and has to be: this tree produces no hash for the nulled document
        # at all, because this tree refuses it.
        #
        # THAT JUSTIFICATION DOES NOT REACH EVERY FIGURE BELOW, AND THE
        # SENTENCE ABOVE CLAIMED IT DID UNTIL 2026-08-24. It covers hashes
        # produced by THIS tree. The one figure the block below DENIES is
        # denied in a sentence about how ``dff95fc`` and ``198a2b5`` hash
        # the NULLED document — and those two DO produce one for it.
        # Re-driven at both commits, at ``JAX_ENABLE_X64=1``, on the
        # document this recipe names: both ACCEPT it and both return one and
        # the same hash, where this commit raises ``TranscriptionError``.
        # The table below that reads ``dff95fc, 198a2b5 -> ACCEPTED, both to
        # the SAME hash`` says exactly that and has all along, which makes
        # this an internal contradiction and not a new measurement. (It is
        # named by what it says rather than by how far down it sits, for the
        # reason the parenthetical about "twelve lines below" gives.)
        #
        # So the rule is narrower than it read: a figure of THIS tree's is a
        # figure of the traced document; a figure of another tree's may be
        # either, and which it is has to be read off the sentence carrying
        # it.
        #
        # The sentence that introduces the harness — *"on a document whose
        # declaration carries ``["dtype", null]``"* — names the NULLED
        # document and the arithmetic below counts hashes of the
        # TRACED one, so as written the recipe named FOUR documents where
        # it meant two — the ``return`` ambiguity below, times which of
        # these two is meant. (That four is not the four VALUES the
        # spelling sweep below collapses to. The sweep is over the traced
        # document alone.)
        #
        # THE ``return`` IS PART OF THE RECIPE AND THIS SENTENCE OMITTED IT
        # UNTIL 2026-08-22, which is a bigger hole than the cell was and the
        # cause of the whole disagreement recorded below. It read "a traced
        # ``any_array((2,), float64, (0.0, 1.0))`` with ``assert_(x >= 0.0)``
        # ... under a ``float64`` outvar aval", and BOTH of the obvious
        # harnesses satisfy that sentence: the declaration's outvar aval is
        # ``float64`` whichever is returned and in either cell, so the clause
        # meant to pin the document does not discriminate at all.
        #
        # THE CELL IS PART OF THE RECIPE AND THIS SENTENCE OMITTED IT UNTIL
        # 2026-08-22. Without ``JAX_ENABLE_X64=1`` jax warns that the
        # requested dtype "is not available, and will be truncated to dtype
        # float32", and a reader following the recipe traces a DIFFERENT
        # document — a fourth equation, an inserted ``convert_element_type``
        # to ``float32``, with the ``ge`` comparison then taking ``float32``
        # operands where the x64=1 one takes ``float64`` and the ``0.0``
        # literal stored as ``<f4`` rather than ``<f8``. Which cell it was
        # traced in is therefore part of the recipe and not a detail of it.
        #
        # THAT SENTENCE SAID SOMETHING ELSE UNTIL 2026-08-22 AND IT WAS
        # WRONG TWICE, both in the direction of putting the difference where
        # it is easiest to look. It read "the outvar aval is not ``float64``
        # at all" and "``float32`` in the declaration where the x64=1 one
        # carries ``float64`` only". Measured here on this commit and on
        # ``844ba48``: the DECLARATION equation is IDENTICAL in the two
        # cells with ``source_info`` stripped — params ``["dtype",
        # "float64"]``, outvar aval ``float64`` — and the jaxpr's own
        # ``outvars[0]`` aval is the same in both cells too, whichever
        # reading of the recipe is taken (``bool`` returning the
        # ``assert_``, ``float64`` returning ``x``). The float32 is
        # downstream of the declaration, not in it. The claim the cell
        # matters survives; the two facts offered for it did not.
        #
        # NO HASH IS QUOTED FOR EITHER CELL, and the reason is not the one
        # this paragraph gave for one day. A pair of literals stood here,
        # inside the paragraph that exists to explain why an earlier literal
        # was deleted, under the heading ``THEY ARE NOT WRONG`` and the claim
        # that an independent re-derivation landing on a DIFFERENT pair was
        # an unexplained "three contexts, three pairs, one recipe". BOTH
        # PAIRS RE-DERIVE, and what separates them is that the recipe above
        # does not say what the harness RETURNS:
        #
        #     return assert_(x >= 0.0)     outvars[0] bool
        #     assert_(x >= 0.0); return x  outvars[0] float64
        #
        # Both documents carry the same declaration, the same comparison and
        # the same ``stelling_assert``; both ARE "a traced
        # ``any_array((2,), float64, (0.0, 1.0))`` with ``assert_(x >= 0.0)``"
        # as the recipe words it. They differ in the jaxpr's OUTVARS and
        # nowhere else, and each returns its own pair of values, identical on
        # this commit and on ``844ba48``, in both cells. The pair that stood
        # here is what returning the ``assert_`` gives; the pair the audit
        # measured is what returning ``x`` gives. Nothing was wrong with
        # either measurement and the recipe was wrong with both.
        #
        # THAT ACCOUNTS FOR TWO OF THE THREE PAIRS THIS CAMPAIGN RECORDED,
        # AND "THE WHOLE OF THE DISAGREEMENT" WAS TOO STRONG FOR IT —
        # withdrawn 2026-08-23. A THIRD pair was written down for this same
        # recipe in a sweep, and it is not what either reading above
        # produces in either cell. It is withdrawn there and nothing in this
        # tree asserts it, so there is no figure here to correct; what is
        # corrected is the claim to have accounted for every recorded pair,
        # which had accounted for two. **Two is what is demonstrable**, and
        # why the third is not is this block's own subject: no document was
        # published beside it, so nobody can now say what was traced. That
        # is the argument for deleting digits, made against a sentence of
        # this comment's own.
        #
        # THE HASH DOES NOT MOVE FOR ANYTHING, WHICH IS WORSE. The earlier
        # version of this paragraph said it did, and that overstates the
        # instability in the direction that makes the recipe look unfixable.
        # Driven here, 120 spellings of the harness above per cell — the
        # dtype as ``"float64"``, ``np.float64``, ``jnp.float64`` and
        # ``np.dtype("float64")``, the bounds as ints, floats and mixed, the
        # comparison's right-hand side as ``0`` and ``0.0``, and five things
        # to return — collapse to FOUR distinct values in each cell, and the
        # partition is exactly by what is returned. Dtype spelling, bounds
        # typing and rhs typing all normalise away. So one written recipe
        # names four documents per cell because of one thing it does not say,
        # and a reader with a digit and no document cannot tell "the guard
        # changed" from "I typed a different harness". Four values for one
        # recipe is three too many, and that is why the digits go.
        #
        # AND THE VARIATIONS ARE REAL AND WERE MISCOUNTED. SIX one-line
        # variations of the harness — dropping the ``return``, a scalar
        # shape, ``(3,)``, bounds ``(-1.0, 1.0)``, the other comparator, a
        # ``float32`` declaration — give six more values in each cell. With
        # the recipe's own two readings that is EIGHT values in each cell,
        # all eight distinct, of which exactly TWO per cell are literals some
        # context has recorded for "this recipe" — the two readings. (The
        # ``float32`` variation is the one that reads the SAME in both cells,
        # which says it again: the document decides, and the cell decides the
        # document.) The commit that dropped the digits said SEVEN variations
        # were really six "because the seventh value is the recipe's own, and
        # it IS two of the six literals". Re-derived: the recipe's own is two
        # values per cell, not one, and both are recorded literals — so the
        # premise was half right, the arithmetic was not, and neither
        # collapses a variation. Six variations, two readings, eight values.
        #
        # So this paragraph is held to the rule the block below states about
        # ``64a0ce8d…``: the document is above, and the hash is not needed.
        # (That rule was "twelve lines below" when this sentence was written,
        # then "the rule this comment block ends on" — which was false as
        # written, the block runs well past it. A rule is named by what it
        # says, not by where it sits.)
        #
        # The PROPERTY below holds either way — driven at both commits, in
        # both cells, under BOTH readings of the recipe, four documents and
        # one verdict each side — while the document and its hash do not, and
        # a recipe a reader cannot re-trace is how a measurement turns into a
        # claim:
        #
        #     dff95fc, 198a2b5          ACCEPTED, both to the SAME hash
        #     this commit               TranscriptionError
        #
        # (``198a2b5`` was labelled "main" here when it was the tip. It is an
        # ancestor of main and not main — ``9cb2c0f`` as this is written, and
        # something else by the time it is read, which is the whole reason
        # the sha is what is pinned and the label is what drifted.)
        #
        # THE PROPERTY IS THE MEASUREMENT AND THE LITERAL WAS NOT. This
        # comment read ``content_hash 64a0ce8d…`` until 2026-08-21, and
        # ``SOUNDNESS.md`` had already dropped that literal from its own
        # copy of this entry on the ground that nobody can reproduce it —
        # so the record and the code disagreed about a figure, in the
        # direction where the code was the one still asserting it. Driven
        # here at both commits, in this worktree, on the document
        # described above: ``dff95fc`` and ``198a2b5`` accept it and
        # return the identical hash, and it is not ``64a0ce8d…``. What a
        # reader can re-derive is the PROPERTY — accepted on both, to one
        # hash, refused here — and that is what this says now. A hash
        # literal without the document that produced it is not a
        # measurement; the document is above, so the hash is not needed.
        #
        # THE REFUSAL IS RIGHT: an absent `dtype` param is a declaration
        # that does not describe its dtype twice, which hand-built IR is
        # allowed to be; a `dtype` param that IS present and ``null`` is a
        # second self-description that contradicts the aval, and
        # `propagate._ieee_any` would have consumed it as
        # ``str(None) == "None"`` — a string naming no ieee format, so the
        # declaration takes the no-float-format arm and gets no subnormal
        # band, which is the same silent misread `b'float64'` produces.
        #
        # It is also the ONE document for which this batch's
        # "byte-identical `content_hash()` across three trees" claim cannot
        # hold, and `CHANGELOG.md` says so where it makes that claim: this
        # commit produces no hash for it at all.
        #
        # THE FIX IS TO CONSTRAIN THE TYPE, not to widen the comparison. A
        # `dtype` param is a NAME — the same fact `_canonical_param_keys`
        # states about a param key — and `propagate._ieee_any` consumes it
        # with ``str()``, which turns ``b'float64'`` into ``"b'float64'"``:
        # a string naming no ieee format, so the declaration silently gets
        # the no-float-format arm and no subnormal band. That the auditor
        # could not drive the three spellings to different VERDICTS is a
        # fact about how much of the haze is re-derived downstream from the
        # aval, not a reason to leave a self-description unchecked.
        #
        # CANONICALIZED FIRST, because this runs BEFORE
        # `_canonical_param_values` — the door's generic pass over the
        # params cannot run until this function has installed the two it
        # owns — so `raw_dtype` here is still exactly as the document
        # handed it over. One read through the door, then a TYPE test on
        # what the door produced; the previous spelling's `isinstance`
        # took the object's word for it and then applied `str.__str__` to
        # whatever answered, which is a raw `TypeError` out of a public
        # constructor for anything that lied.
        try:
            dtype = _canonical(raw_dtype)
        except _NotCanonical as exc:
            exc.path.append(".params['dtype']")
            _refuse_uncanonical(exc, where)
        if type(dtype) is not str:
            # COMPOSED ONLY ON THE FAILING PATH, for the reason
            # `_canonical_param_keys` gives and one more that is this
            # batch's own subject: this message quotes the RAW param, and
            # `repr()` of a document-supplied object is a READ of it. The
            # door's passing path is exactly ONE read per value, which is
            # the whole of what makes a second read impossible to
            # disagree with; a quote composed as an ARGUMENT would make it
            # two on every declaration that has nothing wrong with it.
            _load_check(
                False,
                where,
                f"stelling_any dtype param {_safe_repr(raw_dtype)} is of "
                f"type {_safe_type_name(raw_dtype)}, and a dtype is a "
                f"NAME: it is the second of a declaration's two "
                f"self-descriptions of one declared set, it is compared "
                f"below against the outvar aval dtype "
                f"{_safe_repr(eqn.outvars[0].aval.dtype)}, and "
                f"`propagate._ieee_any` selects the subnormal band by "
                f"reading `str()` of it — so a param of any other type is "
                f"not a disagreement this function can report, it is a "
                f"comparison that never happened",
            )
        _load_check(
            dtype == eqn.outvars[0].aval.dtype,
            where,
            # BOTH QUOTES GUARDED, AND NEITHER CAN RAISE ON THIS TREE —
            # which is a fact about two other guards and not about this
            # message, so it is not the reason they are guarded. `dtype`
            # is an exact `str` by the type test three lines up, and the
            # aval's dtype is an exact `str` or `None` by
            # `Aval.__post_init__`'s own canonicalization. This function
            # does not rest on either having run (see the paragraph on the
            # defensive extent read above), and
            # `tests/test_ir_message_totality.py` measures this very
            # statement escaping raw with the door's leaf reads neutered.
            # It sat 44 lines under a comment claiming every quote in this
            # function was guarded, and was not (audit 0.2.0 B6 audit 4,
            # F2)
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

# and the door's index, which merges this set with the two above it. It is
# built HERE and nowhere else but `_register_stored_type`; see
# :func:`_rebuild_stored_index` for why that is stated rather than left to
# be noticed.
_rebuild_stored_index()
