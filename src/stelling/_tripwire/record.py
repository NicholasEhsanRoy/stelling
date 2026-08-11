# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What the tripwire records, and the arithmetic it checks itself against.

**Pure Python. No jax, and nothing jax-shaped crosses into here.** A
:class:`Finding` holds strings, ints and tuples of them and nothing else,
which is what makes this module testable in a bare interpreter *and* what
makes the xdist payload serialisable. One discipline, two payoffs;
``design/private-jax-boundary.md`` records the first reason and
``PLAN-tripwire.md`` §6 the second.

The two things worth reading before the code:

**The recomputation is the point.** :func:`narrow` reimplements the narrowing
in Python integer arithmetic, from the recorded ``(written, dtype)`` alone,
without going anywhere near jax. Every finding is checked against it before it
is printed, and a disagreement is reported *as a disagreement* rather than as
a finding. That is this tool's analogue of ``_require_valid_refutation``: a
verdict that is replayed before it is reported.

**Attribution is a filter, not a lookup.** The innermost non-jax frame is
often the writer and sometimes only the *caller* of a jax function that wrote
the constant itself — jax's own PRNG mask is exactly that shape. Measured
stacks for both are in :func:`attribute`'s docstring.
"""

from __future__ import annotations

import linecache
from dataclasses import dataclass, field, replace

# name -> (signed, bit width). Spelled out rather than derived from numpy:
# this module must import in an interpreter that has neither jax nor numpy.
# Anything not in here is not range-checked and is counted as such, which is
# the honest handling for float, bool, complex and extended dtypes.
INT_DTYPES: dict[str, tuple[bool, int]] = {
    "int8": (True, 8),
    "int16": (True, 16),
    "int32": (True, 32),
    "int64": (True, 64),
    "uint8": (False, 8),
    "uint16": (False, 16),
    "uint32": (False, 32),
    "uint64": (False, 64),
}

#: A stack frame, as primitives: ``(file, line, func)``.
Frame = tuple[str, int, str]


def dtype_range(dtype: str) -> tuple[int, int] | None:
    """Inclusive ``(lo, hi)`` for an integer dtype name, or None if not one."""
    spec = INT_DTYPES.get(dtype)
    if spec is None:
        return None
    signed, bits = spec
    return (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2**bits - 1)


def jaxpr_dtype(dtype: str) -> str:
    """How a jaxpr PRINTS this dtype: ``int8`` -> ``i8``, ``uint8`` -> ``u8``.

    The reproducer predicts what a user will see, and it predicted
    ``44:int8[]`` while jaxprs print ``44:i8[]``. Measured over the 13 findings
    a real session produced: 13 of 13 reproduce the VALUE and 0 of 13 print the
    predicted text. A prediction a reader can check has to be checkable in the
    direction they will check it.

    Derived from :data:`INT_DTYPES` rather than tabulated, so a dtype added
    there cannot come with a wrong abbreviation. Re-measured on 0.11.0 and
    0.10.2 for all eight names.
    """
    spec = INT_DTYPES.get(dtype)
    if spec is None:
        return dtype
    signed, bits = spec
    return f"{'i' if signed else 'u'}{bits}"


def in_range(value: int, dtype: str) -> bool | None:
    """Whether ``value`` is representable in ``dtype``; None if not an int dtype."""
    bounds = dtype_range(dtype)
    if bounds is None:
        return None
    return bounds[0] <= value <= bounds[1]


def narrow(value: int, dtype: str) -> int | None:
    """The narrowing, recomputed from scratch in Python integer arithmetic.

    This is the independent route. It never touches jax, numpy, or the
    recorded output — it takes the value the user wrote and the dtype it was
    written into, and says what two's-complement truncation makes of it. The
    report compares it against what was *observed* at the hook, and prints the
    disagreement rather than the finding if they differ.

    Returns None for a dtype this module does not model, which is not a
    silence: :class:`Recorder` counts those separately.
    """
    spec = INT_DTYPES.get(dtype)
    if spec is None:
        return None
    signed, bits = spec
    wrapped = value % (2**bits)
    if signed and wrapped >= 2 ** (bits - 1):
        wrapped -= 2**bits
    return wrapped


def arithmetic_sentence(written: int, dtype: str) -> str:
    """The one line a reader checks with a pencil, no instrument involved."""
    spec = INT_DTYPES.get(dtype)
    if spec is None:
        return f"{dtype} is not an integer dtype this tool models"
    signed, bits = spec
    modulus = 2**bits
    if not signed:
        return f"{written} mod 2**{bits} = {written % modulus}"
    unsigned = written % modulus
    if unsigned < 2 ** (bits - 1):
        return f"{written} mod 2**{bits} = {unsigned}, which is < 2**{bits - 1}, so {unsigned}"
    return (
        f"{written} mod 2**{bits} = {unsigned}, which is >= 2**{bits - 1}, "
        f"so {unsigned} - 2**{bits} = {unsigned - modulus}"
    )


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

#: Function names in jax's tracing machinery that open a *new* traced region.
#: A jax frame with one of these names sitting between the candidate writer
#: and the const-fold site means the traced function is jax's own, not the
#: user's — see :func:`attribute`. Supplied by the adapter at install time so
#: that this module never has to know which jax it is talking to; repeated
#: here as the default only so the pure-Python tests have something to drive.
DEFAULT_TRACE_ENTRY_NAMES = frozenset(
    {
        "trace_to_jaxpr",
        "trace_to_jaxpr_nocache",
        "trace_to_jaxpr_dynamic",
        "trace_to_subjaxpr_dynamic",
        "trace_to_jaxpr_final",
    }
)

ORIGIN_USER = "user"
ORIGIN_JAX = "jax"
ORIGIN_UNKNOWN = "unattributed"


def attribute(
    frames: tuple[Frame, ...],
    jax_root: str,
    trace_entry_names: frozenset[str] = DEFAULT_TRACE_ENTRY_NAMES,
) -> tuple[int | None, str]:
    """Pick the writer frame out of a stack, and say whose code wrote it.

    ``frames`` is outermost-first, exactly as :func:`traceback.extract_stack`
    orders it, and has already had the tripwire's own frames removed.

    Returns ``(index_of_writer_or_None, origin)``.

    MEASURED, jax 0.11.0 / 0.10.2, both x64 settings. Two stacks, both ending
    at the same const-fold site, and the difference between them is the whole
    reason this function is not one line:

    ``lambda a: a + 256`` — the user wrote it::

        ...
        10  JAX   partial_eval.py:2290:trace_to_jaxpr_nocache
        11  USER  yours.py:54:<lambda>          <- the writer
        12  JAX   array_methods.py:1532:op
        ...

    ``jax.random.key(0)`` — jax wrote it (``0xFFFFFFFF -> -1``, int32, at
    ``x64=0``; the fire the plan's §1 records over jax's whole test suite)::

        2   USER  yours.py:55:<lambda>          <- only the CALLER
        3   JAX   core.py:258:key
        ...
        17  JAX   partial_eval.py:2290:trace_to_jaxpr_nocache
        18  JAX   threefry2x32.py:73:_threefry_seed   <- the real writer
        ...

    "Innermost non-jax frame" gets the first one right and the second one
    wrong: it would print the user's ``jax.random.key(0)`` line and claim they
    wrote ``4294967295``. The signal that separates them is the trace
    boundary — the frame *immediately inside* the innermost trace entry is the
    function being traced, and in the second stack that function lives in jax.
    So:

    * find the innermost frame whose name is a trace entry;
    * the frame immediately inside it is the traced function, and the writer
      is the innermost non-jax frame at or inside that point;
    * if there is none, jax wrote the constant — ``ORIGIN_JAX``, and the index
      returned is jax's *own* writer (``threefry2x32.py:73`` above), so the
      caller can name it rather than dropping it or pointing at the fold site.

    With no trace entry in the stack at all — a jax that renamed them — this
    falls back to the innermost non-jax frame and reports ``ORIGIN_UNKNOWN``
    if that is missing too. The fallback is deliberately the *lenient* one:
    over-reporting an attribution is visible to the user, who has the quoted
    line and the reproducer to check it with, whereas a filter that silently
    swallowed a real narrowing would not be.
    """
    if not frames:
        return None, ORIGIN_UNKNOWN

    def is_jax(frame: Frame) -> bool:
        return bool(jax_root) and frame[0].startswith(jax_root)

    entry = trace_entry_index(frames, jax_root, trace_entry_names)

    if entry is None:
        for i in range(len(frames) - 1, -1, -1):
            if not is_jax(frames[i]):
                return i, ORIGIN_USER
        return None, ORIGIN_UNKNOWN

    for i in range(len(frames) - 1, entry, -1):
        if not is_jax(frames[i]):
            return i, ORIGIN_USER
    # Every frame inside the traced region is jax's. The traced function is
    # the frame immediately inside the entry, which is jax's own writer —
    # report THAT rather than the fold site, so the suppression names
    # something a reader can go and look at.
    return (entry + 1 if entry + 1 < len(frames) else None), ORIGIN_JAX


def trace_entry_index(
    frames: tuple[Frame, ...],
    jax_root: str,
    trace_entry_names: frozenset[str] = DEFAULT_TRACE_ENTRY_NAMES,
) -> int | None:
    """Index of the innermost jax frame that opens a traced region, or None."""
    for i in range(len(frames) - 1, -1, -1):
        file, _, func = frames[i]
        if func in trace_entry_names and jax_root and file.startswith(jax_root):
            return i
    return None


def user_chain(
    frames: tuple[Frame, ...],
    jax_root: str,
    trace_entry_names: frozenset[str] = DEFAULT_TRACE_ENTRY_NAMES,
) -> tuple[Frame, ...]:
    """The non-jax frames INSIDE the traced region, outermost first.

    §8 asks for the full non-jax chain, and the first version gave literally
    that — every non-jax frame on the stack. Measured under a real pytest run
    that is forty lines of ``runpy``, ``_pytest.runner`` and ``pluggy``
    per finding, on a report whose whole job is to be readable at a glance.
    None of it is about the user's program: the frames that carry the
    cross-module story (``helper.py:3`` resolving from another module) are the
    ones between the trace entry and the site, and they are all of it.

    With no trace entry found, this falls back to the whole non-jax stack —
    the same lenient direction :func:`attribute` takes, and for the same
    reason.
    """
    entry = trace_entry_index(frames, jax_root, trace_entry_names)
    window = frames if entry is None else frames[entry + 1 :]
    return tuple(f for f in window if not (jax_root and f[0].startswith(jax_root)))


def source_line(file: str, line: int) -> str:
    """The user's own line, read with :mod:`linecache`. Empty if unreadable."""
    try:
        return linecache.getline(file, line).rstrip("\n").strip()
    except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
        return ""


# ---------------------------------------------------------------------------
# The finding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One out-of-range integer narrowing, as primitives only.

    ``written`` and ``from_dtype`` are read off the rule's *input*;
    ``became`` and ``to_dtype`` off its *output* and its params. Neither half
    is inferred — both are observed at the site, which is what §10a.2 asks
    for. :attr:`recomputed` is the independent route (:func:`narrow`), and
    :attr:`agrees` is the comparison the report refuses to print a finding
    without.
    """

    file: str
    line: int
    func: str
    written: int
    from_dtype: str
    to_dtype: str
    became: int
    origin: str
    chain: tuple[Frame, ...] = ()
    count: int = 1
    literal_visible: bool = False

    @property
    def key(self) -> tuple[str, int, int, str]:
        """The dedup key the plan names: ``(file, line, written, dtype)``."""
        return (self.file, self.line, self.written, self.to_dtype)

    @property
    def recomputed(self) -> int | None:
        return narrow(self.written, self.to_dtype)

    @property
    def agrees(self) -> bool:
        """Whether the independent recomputation matches what was observed."""
        return self.recomputed == self.became

    @property
    def source(self) -> str:
        return source_line(self.file, self.line)

    @property
    def sort_key(self) -> tuple[str, int, int, str]:
        """§10a.7: stable ordering, never emission order."""
        return (self.file, self.line, self.written, self.to_dtype)

    def as_tuple(self) -> tuple:
        """Primitives only, for the execnet hop. See :meth:`from_tuple`."""
        return (
            self.file,
            self.line,
            self.func,
            self.written,
            self.from_dtype,
            self.to_dtype,
            self.became,
            self.origin,
            tuple(tuple(f) for f in self.chain),
            self.count,
            self.literal_visible,
        )

    @classmethod
    def from_tuple(cls, payload) -> Finding:
        (
            file,
            line,
            func,
            written,
            from_dtype,
            to_dtype,
            became,
            origin,
            chain,
            count,
            literal_visible,
        ) = payload
        return cls(
            file=file,
            line=line,
            func=func,
            written=written,
            from_dtype=from_dtype,
            to_dtype=to_dtype,
            became=became,
            origin=origin,
            chain=tuple((c[0], c[1], c[2]) for c in chain),
            count=count,
            literal_visible=literal_visible,
        )


# The cap exists so that a pathological suite cannot turn a 200-line report
# into a 200,000-line one. It is a cap on DISTINCT findings, after dedup, and
# hitting it is disclosed rather than silent (§6).
DEFAULT_CAP = 200


@dataclass
class Recorder:
    """Counters and findings for one process. Everything here is a number or
    a :class:`Finding`; nothing jax-shaped is stored.

    The counters exist because §8's first disclosure obligation is the
    denominator: *"0 findings over 4,812 narrowing conversions"* is evidence
    and *"0 findings"* is indistinguishable from a hook that never ran.
    """

    #: every entry into the wrapper, folded or not
    invocations: int = 0
    #: the rule returned a constant (it returns None for non-scalars — measured)
    folded: int = 0
    #: folded, integer source, integer target: the population that is range-checked
    int_narrowings: int = 0
    #: folded but not integer-to-integer, so outside what this tool models
    unmodelled: int = 0
    #: raw out-of-range narrowings, before dedup, before the origin filter
    fires: int = 0
    #: fires whose writer is inside jax itself — named and counted, never dropped
    suppressed_jax: int = 0
    #: fires with no attributable frame at all
    unattributed: int = 0
    #: distinct findings dropped because the cap was reached
    dropped_over_cap: int = 0
    #: exceptions raised inside the wrapper's own bookkeeping, swallowed so a
    #: measurement instrument can never break the suite it is measuring
    internal_errors: int = 0
    cap: int = DEFAULT_CAP
    findings: dict = field(default_factory=dict)
    suppressed: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.invocations = 0
        self.folded = 0
        self.int_narrowings = 0
        self.unmodelled = 0
        self.fires = 0
        self.suppressed_jax = 0
        self.unattributed = 0
        self.dropped_over_cap = 0
        self.internal_errors = 0
        self.findings = {}
        self.suppressed = {}

    @property
    def count(self) -> int:
        """Distinct user-attributed findings. What the self-check counts."""
        return len(self.findings)

    def add(self, finding: Finding) -> None:
        bucket = self.findings if finding.origin == ORIGIN_USER else self.suppressed
        existing = bucket.get(finding.key)
        if existing is not None:
            # ``+ finding.count``, not ``+ 1``: this is also the merge path for
            # a worker payload, whose findings arrive already counted.
            bucket[finding.key] = replace(existing, count=existing.count + finding.count)
            return
        if finding.origin == ORIGIN_USER and len(bucket) >= self.cap:
            self.dropped_over_cap += 1
            return
        bucket[finding.key] = finding

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings.values(), key=lambda f: f.sort_key)

    def sorted_suppressed(self) -> list[Finding]:
        return sorted(self.suppressed.values(), key=lambda f: f.sort_key)

    # -- the xdist payload, primitives only ---------------------------------

    def as_payload(self) -> dict:
        return {
            "invocations": self.invocations,
            "folded": self.folded,
            "int_narrowings": self.int_narrowings,
            "unmodelled": self.unmodelled,
            "fires": self.fires,
            "suppressed_jax": self.suppressed_jax,
            "unattributed": self.unattributed,
            "dropped_over_cap": self.dropped_over_cap,
            "internal_errors": self.internal_errors,
            "findings": [f.as_tuple() for f in self.findings.values()],
            "suppressed": [f.as_tuple() for f in self.suppressed.values()],
        }

    def absorb(self, payload: dict) -> None:
        """Merge another process's payload into this one. Deduplicates across
        workers by the same key, summing counts — two workers that each traced
        the same wrap report one finding seen twice, not two findings."""
        for name in (
            "invocations",
            "folded",
            "int_narrowings",
            "unmodelled",
            "fires",
            "suppressed_jax",
            "unattributed",
            "dropped_over_cap",
            "internal_errors",
        ):
            setattr(self, name, getattr(self, name) + int(payload.get(name, 0) or 0))
        for raw in payload.get("findings", ()):
            self.add(Finding.from_tuple(raw))
        for raw in payload.get("suppressed", ()):
            self.add(Finding.from_tuple(raw))
