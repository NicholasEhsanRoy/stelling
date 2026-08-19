# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The only file in stelling permitted to name a private jax module.

It locates the const-fold registry, installs a recording wrapper over the
``convert_element_type`` rule, probes it in both directions, and restores it
by identity.

WHY THE EXEMPTION EXISTS, in one paragraph, because a reader who finds this
file needs the reason before the code. The tripwire attaches to
``const_fold_rules``, keyed on ``convert_element_type_p``. The primitive is
public — ``jax.extend.core.primitives`` — and is the same object; the registry
keyed on it is not exported by any public or ``jax.extend`` module. Measured on
both tested series: seven candidate modules, including
``jax.interpreters.partial_eval``, which imports cleanly and does not carry it.
So there is no route to this hook that does not name a private module, and the
choice was an exemption or no feature. ``design/private-jax-boundary.md``
carries the table and the alternatives that were rejected.

WHAT THE EXEMPTION IS, EXACTLY. It is rule 2 only — ``jax._src`` — and it is
pinned to this one path in both controls (the ``jax-import-hygiene`` hook
filters on the anchored path; ``tests/test_import_hygiene.py`` compares the
repo-relative path). **This file is still subject to rule 1**: it may not spell
``import jax`` or ``from jax``, and it does not. The private module is reached
through :func:`importlib.import_module` with a plain literal name — which is
also what the fail-closed contract needs, since this adapter must *probe* and
report rather than import and die — and public jax, which the self-check needs
for real work, comes through ``stelling._jax_compat`` like everything else in
the package. The exemption bought one private module, not a second jax
boundary.

NOTHING JAX-SHAPED LEAVES THIS MODULE. Every public function here returns a
string, an int, a tuple of primitives, or None. That is this rule's
requirement — the exemption is void the moment the private object is reachable
from a module that does not have it — and it is also the tripwire's own, since
a finding has to survive an xdist process boundary. One discipline, two
payoffs.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import re
import sys

from stelling import _optional
from stelling._tripwire import record

# The private module and the attribute on it. Spelled as data rather than as an
# import statement on purpose: this module must import cleanly with jax absent,
# report ``no-module``, and never raise at import.
PRIVATE_MODULE = "jax._src.interpreters.partial_eval"
REGISTRY_ATTR = "const_fold_rules"

# ---------------------------------------------------------------------------
# Version bounding (PLAN-tripwire.md §5)
#
# BEHAVIOUR IS THE CONTRACT; THE VERSION IS A PRE-FILTER AND A DISCLOSURE. The
# probe in :func:`selfcheck` is the authority, because a version claim is one
# nothing verifies and a patch release can move the rule without moving the
# minor version.
# ---------------------------------------------------------------------------

#: Below this, refuse without probing. 0.4.8 is the release nearest jax commit
#: ``c2fe350455``, which created the line that wraps: below it there is no
#: reason to believe the rule this tool attaches to does what it attaches for.
#:
#: THE 0.5.1 HASH DOES NOT JUSTIFY THIS BOUND, and this comment used to say it
#: did — *"the rule's source is measurably different on 0.5.1 (sha1
#: ``f5f2d0057376``), so probing there would be probing a function this tool
#: has never read"*. 0.5.1 is ABOVE 0.4.8, so the floor does not exclude it and
#: it IS probed. The differing hash is a disclosure that the rule's source
#: moves inside the range this tool will arm on — which is exactly why §5 makes
#: the probe the authority and records the hash rather than gating on it — and
#: not an argument for where the floor sits.
#:
#: THE PRECISION IS NOT MEASURED AND SAYS SO. Nothing here has been run on
#: 0.4.8, and stelling's own floor is ``jax>=0.10`` anyway
#: (``pyproject.toml``), so in every environment stelling supports this bound
#: is inert. It is a refusal boundary, not a support claim.
_FLOOR = (0, 4, 8)

#: The series with a CI lane, from the one place that fact is kept.
_TESTED = _optional.TESTED_JAX_SERIES

#: sha1[:12] of the rule's source, keyed on the EXACT jax release it was read
#: on. RECORDED, NOT GATED ON: :func:`version_check` and :func:`selfcheck`
#: never consult this table, so a cosmetic edit upstream cannot disable the
#: tool — the tool arms against whatever the map says. What the map buys is
#: that a changed hash in the status line, in the canary and in the test is
#: attributable to a release, without re-running anything.
#:
#: A MAP AND NOT A SET, and this is the whole value of the design.
#:
#: * A set of known-good hashes GOES GREEN ON A REVERT. If jax 0.12 restored
#:   0.11.0's spelling, a set would say "known" and mean nothing — the rule
#:   had just changed under the tool twice.
#: * A set cannot say WHICH release carries which rule, which is the first
#:   question anyone asks when the canary fires.
#: * A set cannot express "this release has never been read". A RELEASE
#:   missing from this map is a FAILURE in ``tests/test_tripwire_arm.py``
#:   and a loud line in the canary, and it has to be: the remedy is that
#:   somebody reads the rule and writes down what moved. (A nightly or an
#:   rc is *not* a release, cannot be given a row, and is asserted to be in
#:   the never-read state instead — :func:`is_release`.)
#:
#: THE COST OF "MISSING IS A FAILURE", STATED SO IT IS NOT A SURPRISE: the
#: day jax ships any release with no row here, the floating ``test-jax``
#: lane goes red, and so does ``test-jax-0-10`` if a 0.10.3 ever ships,
#: since that lane pins the SERIES and resolves to the newest 0.10.x. The
#: remedy in both cases is the same and is deliberately manual: read the
#: rule, diff it, add a row.
#:
#: HOW SOON THAT RED ARRIVES IS DECIDED BY TRIGGERS, AND THIS USED TO SAY
#: "within a working day", WHICH NO TRIGGER DELIVERS. Read off the two
#: workflows:
#:
#: * ``ci.yml`` — which is the only file that runs ``test-jax`` — has
#:   ``on: push: branches: [main]`` and ``on: pull_request``, and NO
#:   ``schedule:``. So the red arrives on the next push to ``main`` or the
#:   next pull-request event, whenever that is. On a quiet week it is a
#:   quiet week; there is no clock behind it.
#: * ``nightly-jax-canary.yml`` DOES have a clock (``cron: "17 4 * * *"``),
#:   and it does not go red on this. Driven, with the version reported as a
#:   release that has no row: the canary prints the loud
#:   ``HAS NEVER BEEN READ`` line to stdout and to the step summary and
#:   **exits 0** — by design, argued in ``_hash_row``. Its ``control`` leg
#:   installs ``-e ".[jax]"`` and so is the leg that meets a new release
#:   first; its ``nightly`` leg installs a dev build, which is not a release
#:   (:func:`is_release`), so the row check there is carved out by
#:   construction.
#:
#: So: a daily job SEES a rowless release within a day and says so without
#: failing; the failing signal is event-triggered. Both are true and they
#: are different sentences.
#:
#: KEYED ON THE EXACT RELEASE, NOT THE SERIES, and this incident is the
#: argument: 0.11.0 and 0.11.1 are one series carrying two different rule
#: sources. ``_optional.TESTED_JAX_SERIES`` stays keyed on the series
#: because it is a claim about which series has a CI lane, which is a
#: different fact — see :data:`_TESTED` above.
#:
#: EVERY ROW NAMES WHAT MOVED. Adding a row by copying an observed hash in
#: defeats the instrument; the entry is a record that someone diffed the
#: rule against the nearest row and can say what changed.
#:
#: 0.5.1 HAS NO ROW, deliberately, even though :data:`_FLOOR`'s comment
#: above records a hash for it (``f5f2d0057376``) and :data:`_KNOWN_RULE`
#: records its name. A row here is not a note — it is a PIN the suite
#: enforces, and no lane can run 0.5.1: ``pyproject.toml`` floors stelling
#: at ``jax>=0.10``, so a 0.5.1 row would be a pin nothing ever checks,
#: which is the shape of claim this repository keeps having to withdraw.
#: The disclosure belongs where it already is, in the floor's comment. A
#: 0.5.1 environment reports ``never-read``, which is the true statement
#: about a release this table has no enforced reading of.
_KNOWN_HASHES: dict[str, str] = {
    # 0.10.2 and 0.11.0 are byte-identical here, independently re-derived
    # twice. 0.10.2's hash is a prior measurement carried forward: there is
    # no 0.10.2 interpreter in this repository's CI, only the 0.10 series
    # lane, which resolves to whatever 0.10.x is newest.
    "0.10.2": "c808b3001114",
    "0.11.0": "c808b3001114",
    # jax 803de7b08 (2026-08-11), released in 0.11.1, changed the rule's
    # scalar test by one line:
    #     -      and not np.shape(c)
    #     +      and not out_aval.shape
    # Read and measured before this row was written. It is
    # semantics-preserving FOR THIS TOOL: the two spellings differ only when
    # ``np.shape(t.get_const()) != t.aval.shape``, and over a
    # ``DynamicJaxprTrace`` that cannot happen — ``_new_const`` binds value
    # and aval together, and all three registered fold rules preserve the
    # pairing. Measured in the qualification that preceded this row at
    # 122,672 const-fold invocations per version with zero disagreements,
    # over a combination table byte-identical across the two versions —
    # carried forward here, not re-derived. What WAS re-derived for this
    # row: both hashes, the one-line diff above, and that the rule's NAME
    # did not move.
    "0.11.1": "522706b62a10",
}

#: The rule's name on 0.5.1, 0.10.2, 0.11.0 and 0.11.1. Also recorded, not
#: gated on. Unlike the source hash this has not moved on any release read so
#: far, which is why it is one string and not a map.
_KNOWN_RULE = "_convert_elt_type_folding_rule"

#: Frames that open a new traced region. See :func:`record.attribute` for the
#: two measured stacks that make this necessary; the names live here because
#: this is the module that is allowed to know things about jax's internals.
TRACE_ENTRY_NAMES = record.DEFAULT_TRACE_ENTRY_NAMES

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")

#: A RELEASE, and therefore something :data:`_KNOWN_HASHES` can have a row
#: for. This is PEP 440's **final release** — see :func:`is_release` for what
#: that MEANS and why the meaning is the definition.
#:
#: IT USED TO BE ``^\d+\.\d+\.\d+\Z`` — a bare ``X.Y.Z`` — AND THAT WAS
#: MEASURABLY TOO NARROW. jax has shipped a release that is not bare
#: ``X.Y.Z``: ``0.9.0.1``, uploaded 2026-02-05, two files, not yanked, read
#: off PyPI's JSON API (192 of jax's 195 versions are bare ``X.Y.Z``; the
#: other three are ``0.0``, ``0.1`` and ``0.9.0.1``). Under the old pattern
#: such a release is not a release, so it lands in ``never-read`` and the
#: assertion in ``tests/test_tripwire_arm.py`` that demands a row for it is
#: skipped. Driven, on real jax 0.11.1 (whose rule really has moved) with the
#: version string reported as ``0.11.1.1``: the tree AS MERGED at ``3482822``
#: PASSED, and ``fb646b4`` — the pre-merge tree, a single ``_KNOWN_HASH``
#: constant and no shape carve-out at all — FAILED. That is coverage the
#: 0.11.1 merge lost, not coverage it never had. With the pattern below it
#: fails again, which is the point.
#:
#: THE GRAMMAR, clause by clause, and every clause is PEP 440's own — written
#: out rather than delegated to ``packaging`` because stelling's core is
#: zero-dependency and may not import it:
#:
#: * ``v?`` — the leading ``v`` PEP 440 tolerates.
#: * ``(?:[0-9]+!)?`` — the epoch. **ACCEPTED, deliberately, and here is the
#:   argument.** An epoch does not make a version mutable or unpublishable:
#:   ``1!0.12.0`` on PyPI is exactly as immutable as ``0.12.0``, and
#:   ``importlib.metadata`` reports the string verbatim, so a row keyed on it
#:   can be looked up forever — which is the whole test. The direction of the
#:   error decides the tie: accepting a shape jax never ships costs nothing
#:   (there is no such release to be red about), while rejecting a shape jax
#:   does ship costs exactly the silence this pattern was just widened to
#:   remove. Measured: zero of jax's 195 PyPI versions carry an epoch, so
#:   this clause is inert today and is here for its direction.
#: * ``[0-9]+(?:\.[0-9]+)*`` — the release segment, at ANY number of
#:   components. This is the clause the bare-``X.Y.Z`` pattern did not have.
#: * the post-release group — ``.postN`` and PEP 440's other spellings
#:   (``rev``/``r``, ``-``/``_``/``.`` separators) plus the implicit ``-N``
#:   form. A post-release is a final release: it names a published, immutable
#:   wheel, so a row can name it.
#:
#: AND NOTHING ELSE, because the pattern is anchored at both ends: a
#: pre-release (``0.12.0rc1``), a dev release (``0.11.2.dev20260817``) and a
#: local version (``0.11.1+cuda``) all fail it, which is the point —
#: :func:`is_release` says why. ``\Z`` and not ``$``, because ``$`` also
#: matches before a trailing newline and "and nothing else" would then not be
#: true.
_FINAL_RELEASE_RE = re.compile(
    r"""^
    v?                              # PEP 440 tolerates a leading 'v'
    (?:[0-9]+!)?                    # epoch
    [0-9]+(?:\.[0-9]+)*             # release segment, ANY number of parts
    (?:                             # optional post-release, and only that
        -[0-9]+                     #   implicit post: 1.0-1
      | [-_.]?(?:post|rev|r)[-_.]?[0-9]*
    )?
    \Z""",
    re.VERBOSE,
)

# Module state. One process arms once; ``install`` refuses to double-wrap and
# ``restore`` refuses to clobber someone else's patch.
_installed: dict = {}

#: Bumped once per :func:`selfcheck`, and used as the probe's input length.
#: jax's trace cache is process-wide, so a probe that traced the same avals
#: twice would reach the const-fold site zero times the second time. Measured.
_probe_seq: int = 0


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """``(major, minor, micro)``, or None if unparseable.

    Survives nightlies (``0.4.36.dev20240101``) by reading the leading three
    components and ignoring the rest. **An unparseable version means probe
    anyway** — §5 — so this returning None is not a refusal.
    """
    match = _VERSION_RE.match(text or "")
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def is_release(text: str | None) -> bool:
    """Whether a jax version string names an IMMUTABLE PUBLISHED VERSION —
    PEP 440's **final release**.

    THE MEANING IS THE DEFINITION, and the definition is not "three
    components". A key of :data:`_KNOWN_HASHES` is a string a future reader
    must be able to look up and get the same wheel: that is what makes the
    row a record rather than a note. PEP 440 already names that class — a
    release segment, an optional epoch, an optional post-release — and this
    predicate is that class and nothing more.

    WHAT IS EXCLUDED, AND WHY EACH ONE:

    ``0.11.2.dev20260817`` (dev release)
        names a tree that is rebuilt under the same name, so a hash written
        down against it is a hash of something else tomorrow. This is what
        the ``nightly`` job of ``.github/workflows/nightly-jax-canary.yml``
        installs.
    ``0.12.0rc1`` (pre-release)
        is a candidate, superseded by the release it is a candidate for; the
        rule this tool records is a fact about the release, and pinning the
        candidate would pin a reading nobody will repeat.
    ``0.11.1+cuda`` (local version)
        is a local build of a public release. PEP 440 says local versions are
        not published to an index, so the string identifies nothing anyone
        else can fetch.

    All three land in the ``never-read`` state, and that is what
    ``tests/test_tripwire_arm.py`` asserts of them — rather than that they
    have a row, which would redden a nightly lane for a fact nobody can act
    on.

    WHAT IS NOT EXCLUDED — and this is the correction. ``0.9.0.1`` is a jax
    release (2026-02-05, on PyPI, not yanked) and a bare-``X.Y.Z`` predicate
    called it a non-release, which sent a real published wheel into the
    never-read carve-out and silenced the row check for it. Component COUNT
    was never the question; immutability is. A published wheel is immutable
    whatever its component count, and a nightly is not whatever its component
    count. See :data:`_FINAL_RELEASE_RE` for the grammar, the measurement and
    the epoch decision.

    It says nothing about whether such a jax is supported —
    :func:`version_check` and :func:`selfcheck` do not consult it, and the
    tool arms on nightlies exactly as before. It is also a DIFFERENT question
    from the one :func:`_parse_version` answers: that one reads three leading
    integers to compare against :data:`_FLOOR` and is deliberately lenient
    about everything after them, because an unparseable version means probe
    anyway. This one decides whether a row may name the string at all.
    """
    return bool(_FINAL_RELEASE_RE.match(text or ""))


def jax_version() -> str | None:
    """The installed jax version, without importing jax."""
    return _optional.version("jax")


def jax_root() -> str:
    """The directory prefix that marks a frame as jax's own.

    A string, and only ever a string: :mod:`record` filters frames by it and
    must not learn what jax is. Empty when jax is not importable, which makes
    :func:`record.attribute` treat every frame as the user's — lenient, and
    unreachable in practice since nothing gets that far without jax.
    """
    try:
        module = importlib.import_module("jax")
    except ImportError:
        return ""
    path = getattr(module, "__file__", None)
    return os.path.dirname(path) + os.sep if path else ""


def _registry():
    """The registry dict, or None. Never raises, never leaves this module."""
    try:
        module = importlib.import_module(PRIVATE_MODULE)
    except ImportError:
        return None
    return getattr(module, REGISTRY_ATTR, None)


def _primitive():
    """``convert_element_type_p``, from the PUBLIC surface that exports it."""
    from stelling._jax_compat import jax as _jax  # public jax, via the boundary

    importlib.import_module("jax.extend.core.primitives")
    return _jax.extend.core.primitives.convert_element_type_p


def locate() -> str:
    """Report whether the const-fold registry is where the tripwire expects it.

    A code from the tripwire's fail-closed vocabulary, never an exception and
    never a jax object:

    ``no-module``
        jax is not installed. Static checking is unaffected by this.
    ``no-registry``
        jax is installed and the private module or its registry is not where
        this expects it — a series moved it, which is what a tool keyed on a
        private surface is supposed to survive by refusing rather than by
        guessing.
    ``no-entry``
        the registry is there and has no rule for ``convert_element_type``.
        Distinguishable from ``no-registry`` because the registry's size is
        itself a measured fact — 3 entries on both tested series.
    ``located``
        the registry is there and the rule is in it.

    Codes are stable and greppable.
    """
    if not _optional.available("jax"):
        return "no-module"
    registry = _registry()
    if registry is None:
        return "no-registry"
    try:
        primitive = _primitive()
    except Exception:  # noqa: BLE001 - a probe may not raise
        return "no-registry"
    return "located" if primitive in registry else "no-entry"


def registry_size() -> int | None:
    """How many rules the registry holds. 3 on both tested series — the fact
    that tells ``no-registry`` (nothing there) from ``no-entry`` (a registry
    that no longer keys our primitive)."""
    registry = _registry()
    return None if registry is None else len(registry)


def rule_hash() -> str | None:
    """sha1[:12] of the installed rule's source, or None if unreadable.

    Recorded in the status so a change upstream is *visible*. Never gated on:
    refusing to arm because someone reflowed a comment is the failure mode §5
    names explicitly.
    """
    registry = _registry()
    if registry is None:
        return None
    try:
        primitive = _primitive()
        rule = _installed.get("original") or registry.get(primitive)
        if rule is None:
            return None
        return hashlib.sha1(inspect.getsource(rule).encode()).hexdigest()[:12]
    except Exception:  # noqa: BLE001 - a disclosure may not raise either
        return None


def rule_name() -> str | None:
    """The installed rule's ``__name__``, for the same reason as :func:`rule_hash`."""
    registry = _registry()
    if registry is None:
        return None
    try:
        primitive = _primitive()
        rule = _installed.get("original") or registry.get(primitive)
        return getattr(rule, "__name__", None)
    except Exception:  # noqa: BLE001
        return None


def known_hash() -> str | None:
    """The hash :data:`_KNOWN_HASHES` records for the RUNNING release, or
    ``None`` — which means *this jax has never been read*, not "the rule is
    fine".

    Looked up on the exact version string, so a nightly
    (``0.11.2.dev20260817``) is a miss by construction: a dev build is not a
    release and cannot be given a row. :func:`report.render_status` and the
    canary both distinguish that miss from a mismatch; nothing gates arming
    on either.
    """
    return _KNOWN_HASHES.get(jax_version() or "")


def version_check() -> tuple[str, str]:
    """``(code, disclosure)``. ``code`` is ``"ok"``, ``"below-floor"``, or
    ``"untested"``; the disclosure is a sentence for the report.

    Above the tested range this arms anyway and says so. Refusing on every new
    jax release makes the tool useless the day jax ships — §5.
    """
    text = jax_version() or "unknown"
    parsed = _parse_version(text)
    if parsed is None:
        return "ok", (
            f"jax {text}: version unparseable, so the probe is the only "
            f"authority here. Tested against {', '.join(_TESTED)}."
        )
    if parsed < _FLOOR:
        floor = ".".join(str(n) for n in _FLOOR)
        return "below-floor", (
            f"jax {text} is below the tripwire's floor of {floor}, where the "
            "const-fold rule's source is measurably different. Refusing to "
            "probe rather than guessing."
        )
    if _optional.jax_series_tested(text):
        return "ok", f"jax {text}, a series with a CI lane."
    return "untested", (
        f"tested against {', '.join(_TESTED)}, running against {text}. Armed "
        "anyway — the probe below is the contract, the version is a "
        "disclosure."
    )


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

# THIS FILE, and only this file. Stripped before attribution so the wrapper
# never appears in its own report.
#
# NOT the whole ``_tripwire`` directory, which is what it was first and which
# was wrong in a way that mattered: ``_probe.py`` lives next door, and
# stripping the package would have removed the self-check's own writer frame,
# sent its finding to the suppressed bucket, and made ``selfcheck`` report
# ``not-invoked`` on a perfectly live hook. Measured, not reasoned about — it
# is what the first run of the self-check did.
_OWN_FILE = os.path.abspath(__file__)


def _stack(skip: int) -> tuple[record.Frame, ...]:
    """The current stack as primitive tuples, outermost first, without our own.

    Hand-walked rather than :func:`traceback.extract_stack` because that reads
    and caches every source file in the stack; this runs only on a fire, but a
    suite with many fires should not pay for it repeatedly.
    """
    frames: list[record.Frame] = []
    frame = sys._getframe(skip)
    while frame is not None:
        code = frame.f_code
        filename = code.co_filename
        if os.path.abspath(filename) != _OWN_FILE:
            frames.append((filename, frame.f_lineno, code.co_name))
        frame = frame.f_back
    frames.reverse()
    return tuple(frames)


def _int_or_none(value) -> int | None:
    """A Python int from a folded constant, or None.

    MEASURED, and the shape check is a CHEAP EARLY RETURN rather than a crash
    guard. The rule is invoked with NON-SCALAR constants too (it returns None
    for them, so nothing folds), and ``int()`` on those raises ``TypeError:
    only 0-dimensional arrays can be converted to Python scalars`` — re-driven
    on ``np.ndarray`` and ``jax.Array`` alike.

    This used to say a wrapper without the check "would crash the user's
    trace on the first array constant it met". It would not, twice over: the
    ``except`` two lines below catches ``TypeError`` itself, and the wrapper's
    own ``except Exception`` catches anything that escaped and counts it in
    ``internal_errors``. What the check buys is not safety but silence — no
    raised-and-caught exception per non-scalar constant, and no
    ``internal_errors`` count for something entirely ordinary.
    """
    if value is None:
        return None
    if getattr(value, "shape", ()) != ():
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


# Per-THREAD stack of gate fire counters. Each _pipeline invocation pushes
# a 0; the wrapper increments stack[-1]; _pipeline pops and reads its own
# count. Empty when no gate is active — selfcheck and session-wide
# accumulation go only to recorder.fires.
#
# Thread-local because the stack is POSITIONAL: push/increment/pop must be
# on the same thread to maintain LIFO semantics. A shared list under a
# ThreadPoolExecutor would let thread A's narrowing increment thread B's
# counter (the top of a shared stack is whoever pushed last, not whoever
# is tracing). JAX tracing is single-threaded, but a test harness in a
# ThreadPoolExecutor is not JAX's to forbid.
import threading as _threading

_gate_fire_local = _threading.local()


def _gate_fire_stack() -> list[int]:
    """The current thread's gate-fire stack (created on first access)."""
    try:
        return _gate_fire_local.stack
    except AttributeError:
        _gate_fire_local.stack = []
        return _gate_fire_local.stack


def evict_trace_caches() -> str:
    """Empty jax's trace caches so the NEXT trace re-traces what they hold.

    ``evicted``
        the caches are gone; every jit body the next trace enters will be
        traced again, under the instrument.
    ``no-module``
        jax is not importable, so there is nothing to instrument anyway.
    ``no-clear-caches``
        this jax has no ``clear_caches``. Nothing was evicted, so a caller
        that needs COMPLETE observation has not got it and must say so.
    ``unexpected:<ExcType>``
        it raised. Same conclusion.

    WHY EVICTION AND NOT DETECTION, in the one paragraph a reader needs before
    the cost. jax's trace cache is keyed on the jitted callable and its avals,
    so a ``@jax.jit`` helper that some earlier trace already warmed is REPLAYED
    rather than traced: the const-fold rule never runs over its body and the
    gate sees a clean zero for a region it did not watch. Detecting that from
    the outside was measured and does not work -- ``jax.jit(f, inline=True)``
    hides the replay and leaves NO nested jaxpr in the enclosing jaxpr to
    detect it by, and jax publishes no per-jit trace counter on a public
    surface (``_cache_size``/``_clear_cache`` are private; ``clear_cache()`` is
    public but clears rather than reports and cannot be enumerated;
    ``jax.explain_cache_misses`` logs MISSES, and the state that matters here
    is a HIT). Emptying the cache makes the observation complete by
    construction instead -- within the bounds of the next paragraph -- and
    needs no detector to be right.

    WHAT "COMPLETE" DOES NOT COVER, because the word invites more than it
    earns. This empties JAX'S caches: a value narrowed into a memo jax does
    not own survives it, and three constructs measured on jax 0.11.0 return
    VERIFIED with 0 fires through it -- ``jax.extend.core.jaxpr_as_fun`` over
    a saved jaxpr, a user ``functools.lru_cache`` holding an eagerly narrowed
    value, and ``jax.closure_convert``, a public jax API that traces at setup
    and hoists the narrowed constant. And jax's cache is PROCESS-GLOBAL while
    the gate's fire counter is per-thread, so the window between this call
    and the trace it protects is not atomic: measured 0/400 wrong VERIFIED
    single-threaded and 247/400 with four threads re-warming the same jitted
    helper (399/400 before this existed). Single-threaded, against jax's own
    caches, it is complete; outside that it is an improvement.

    WHAT IT COSTS, because it is a process-global side effect and the caller
    is entitled to know. ``jax.clear_caches()`` also drops the CALLER's
    compiled functions. The call itself SCALES WITH HOW MANY JITTED
    FUNCTIONS ARE LIVE, so "populated" is not a number: measured on jax
    0.11.0 (median of 12), 0.049ms empty, 1.4ms with one live jit, 8.3ms
    with ten, and 41.8ms (39.6-48.3) with fifty. On top of that the next
    call to a jitted function the caller still wanted pays its whole
    trace-and-compile again -- 18ms for a trivial one, 330ms for a
    200-primitive chain. This runs only when the tripwire is ARMED, which is
    opt-in, and ``docs/overflow-tripwire.md`` prices it there.

    Nothing jax-shaped leaves this function: it returns a string.
    """
    try:
        from stelling._jax_compat import jax as _jax  # public jax, via the boundary
    except Exception:  # noqa: BLE001
        return "no-module"
    clear = getattr(_jax, "clear_caches", None)
    if clear is None:
        return "no-clear-caches"
    try:
        clear()
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"
    return "evicted"


def _make_wrapper(original, recorder: record.Recorder, jaxroot: str):
    """Build the recording wrapper. Nothing in here may raise into a trace.

    ``*args, **kwargs`` rather than the measured ``(consts, params,
    out_avals)``. jax calls the rule positionally today; a release that added
    a keyword would make a fixed signature raise ``TypeError`` **inside the
    user's trace**, and a measurement instrument does not get to break the
    thing it measures. ``arm()``'s self-check catches that and disables the
    tool either way — that is the fail-closed floor — but delegating verbatim
    means a purely additive change upstream costs nothing at all.
    """

    def stelling_const_fold_probe(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            recorder.invocations += 1
            if result is None:
                return result
            recorder.folded += 1

            consts = (args[0] if args else kwargs.get("consts")) or ()
            params = (args[1] if len(args) > 1 else kwargs.get("params")) or {}

            written = _int_or_none(consts[0] if consts else None)
            became = _int_or_none(result[0] if result else None)
            from_dtype = str(getattr(consts[0], "dtype", "")) if consts else ""
            to_dtype = str(params.get("new_dtype", ""))

            if (
                written is None
                or became is None
                or from_dtype not in record.INT_DTYPES
                or to_dtype not in record.INT_DTYPES
            ):
                recorder.unmodelled += 1
                return result

            recorder.int_narrowings += 1
            if record.in_range(written, to_dtype):
                return result

            recorder.fires += 1
            stack = _gate_fire_stack()
            if stack:
                stack[-1] += 1
            frames = _stack(2)
            index, origin = record.attribute(frames, jaxroot, TRACE_ENTRY_NAMES)
            if origin == record.ORIGIN_JAX:
                recorder.suppressed_jax += 1
            elif origin == record.ORIGIN_UNKNOWN:
                recorder.unattributed += 1

            if index is None:
                file, line, func = (frames[-1] if frames else ("<unknown>", 0, "?"))
            else:
                file, line, func = frames[index]
            text = record.source_line(file, line)
            # Recorded whatever the origin: §10a.9 — a silent filter is
            # indistinguishable from a blind instrument, so a suppressed fire
            # is named, counted, and kept in its own bucket rather than
            # dropped. :meth:`record.Recorder.add` routes it by origin.
            recorder.add(
                record.Finding(
                    file=file,
                    line=line,
                    func=func,
                    written=written,
                    from_dtype=from_dtype,
                    to_dtype=to_dtype,
                    became=became,
                    origin=origin,
                    chain=record.user_chain(frames, jaxroot, TRACE_ENTRY_NAMES),
                    literal_visible=str(written) in text,
                )
            )
        except Exception:  # noqa: BLE001
            # An instrument that breaks the suite it measures is worse than no
            # instrument. Counted, disclosed in the report, never raised.
            try:
                recorder.internal_errors += 1
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                pass
        return result

    return stelling_const_fold_probe


def install(recorder: record.Recorder) -> str:
    """Wrap the rule. Returns a status code; never raises, never leaks a jax object.

    ``already-armed`` if this process already installed one — arming twice must
    not double-wrap. A second wrapper over the first sees every invocation the
    first does, so the gate counter, which is module state and belongs to no
    recorder, counts each fire twice whoever owns the wrappers; a RECORDER
    counts twice only when both wrappers were handed the same one, which is
    what a re-arm would do and is not what the orphaned probe in
    :func:`restore` produced. Both cases are measured in
    ``tests/test_tripwire_arm.py``.
    """
    code = locate()
    if code != "located":
        return code
    if _installed:
        return "already-armed"
    try:
        registry = _registry()
        primitive = _primitive()
        original = registry[primitive]
        wrapper = _make_wrapper(original, recorder, jax_root())
        registry[primitive] = wrapper
        _installed.update(
            {"registry": registry, "primitive": primitive,
             "original": original, "wrapper": wrapper, "recorder": recorder}
        )
    except Exception as exc:  # noqa: BLE001
        _installed.clear()
        return f"unexpected:{type(exc).__name__}"
    return "installed"


def restore() -> str:
    """Put the original rule back, **by identity**.

    ``not-armed``
        nothing to restore.
    ``foreign-patch``
        something else replaced our wrapper after we installed it. Say so
        rather than silently clobbering it — §4.
    ``restored``
        done.

    A PENDING :func:`detach` IS FIXED UP FIRST, and that is not tidying. The
    pair ``detach``/``reattach`` saves whatever the registry held and puts it
    back, and what it held may be THIS WRAPPER. Retiring the wrapper makes
    that saved entry stale: a later ``reattach()`` would then reinstall a
    probe no ``_installed`` record owns, so ``is_armed()`` says no, ``rule_hash``
    and ``rule_name`` read *stelling's own wrapper* as if it were jax's rule,
    the next ``arm()`` wraps the wrapper, and the state persists for the life
    of the interpreter. That is not hypothetical: ``detach("bypass")`` ->
    ``disarm()`` -> ``reattach()`` is exactly the sequence the §4 foreign-patch
    test drives, and it left every later ``arm()`` in the process reporting
    ``hash_state == "changed"`` against a jax whose rule had not moved.

    WHAT THE SECOND WRAPPER COSTS, MEASURED, because the first account of this
    said it doubled the counts and it does not. The wrap is real: with an
    orphan left behind, the live registry entry is stelling's wrapper around
    stelling's wrapper around jax's rule -- depth 2, measured. But each
    wrapper closes over the recorder it was installed WITH, so the orphan
    writes into a dead one: on jax 0.11.0 the live recorder reported the same
    ``int_narrowings``, ``fires`` and finding count on a polluted process as
    on a clean one, and ``selfcheck()`` passed either way. What doubles is the
    GATE counter, which is thread-local module state no recorder owns and
    which both wrappers increment: ``_pop_gate()`` returns 2 where it should
    return 1, and that is the N in ``preconditions.check()``'s ``trace
    unfaithful: N integer narrowing(s)``. Measured clean 1 / polluted 2 before
    this fix, 1 / 1 after. The refusal does not turn on the magnitude -- a
    wrapper that fired twice fired at least once -- so what the doubling
    corrupted is a number an operator reads, beside a ``rule_name`` and a
    ``hash_state`` that describe stelling's own probe as jax's rule.

    ``restore`` is the operation that invalidates the saved entry, so it is
    the operation that must correct it: once the wrapper is gone, the value
    that stands in its place is the original this record was holding. Doing
    it here rather than in ``reattach`` keeps the fix at the point where the
    fact changes; ``reattach`` cannot tell a stale entry from a live one.
    """
    if not _installed:
        return "not-armed"
    registry = _installed["registry"]
    primitive = _installed["primitive"]
    wrapper = _installed["wrapper"]
    original = _installed["original"]
    if _detached and _detached.get("entry") is wrapper:
        _detached["entry"] = original
    current = registry.get(primitive)
    if current is not wrapper:
        _installed.clear()
        return "foreign-patch"
    registry[primitive] = original
    _installed.clear()
    return "restored"


#: Saved state for :func:`detach`. Separate from ``_installed`` because
#: detaching is deliberately something that happens *to* an armed tripwire.
#: :func:`restore` reads it -- it rewrites a saved ``entry`` that is the
#: wrapper it is retiring, so that ``detach`` -> ``disarm`` -> ``reattach``
#: cannot leave an orphaned probe as the live rule.
_detached: dict = {}


def detach(mode: str) -> str:
    """Break the tripwire the way a jax release would. Returns a status code.

    THIS IS SHIPPED CODE WITH A TEST-SUPPORT PURPOSE, and it is here rather
    than in ``tests/`` for one reason: rule 2 bans naming the private jax
    module in ``tests/`` with no exemption, so a test that reached into the
    registry to break it would have to name what only this file may name. The
    alternative — asserting the fail-closed contract instead of driving it —
    is the shape of control this repository has been burned by.
    ``design/private-jax-boundary.md`` records the rule; ``PLAN-tripwire.md``
    §9 records the two rows this exists for.

    ``mode="entry"``
        remove the rule the tripwire keys on. :func:`locate` then answers
        ``no-entry``, which is the "anchor removed" row.
    ``mode="bypass"``
        put the original rule back under an armed tripwire, so the wrapper is
        attached and never invoked. That is what a version bump actually
        produces, and what a presence check misses.

    :func:`reattach` undoes either. Nothing here leaks a jax object: both
    arguments and both return values are strings.
    """
    registry = _registry()
    if registry is None:
        return "no-registry"
    try:
        primitive = _primitive()
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"
    if _detached:
        return "already-detached"
    current = registry.get(primitive)
    _detached.update({"registry": registry, "primitive": primitive, "entry": current})
    if mode == "entry":
        registry.pop(primitive, None)
        return "detached"
    if mode == "bypass":
        registry[primitive] = _installed.get("original", current)
        return "detached"
    _detached.clear()
    return f"unknown-mode:{mode}"


def reattach() -> str:
    """Undo :func:`detach`. ``not-detached`` if there is nothing to undo."""
    if not _detached:
        return "not-detached"
    registry = _detached["registry"]
    primitive = _detached["primitive"]
    entry = _detached["entry"]
    if entry is None:
        registry.pop(primitive, None)
    else:
        registry[primitive] = entry
    _detached.clear()
    return "reattached"


def is_armed() -> bool:
    """Whether the wrapper this process installed is still the live entry."""
    if not _installed:
        return False
    return _installed["registry"].get(_installed["primitive"]) is _installed["wrapper"]


def live_check() -> str:
    """Whether the tripwire is STILL live, and if not, which way it went.

    :func:`is_armed` answers only yes or no, and the two nos are different
    failures a user has to be told apart:

    ``armed``
        our wrapper is still the live registry entry.
    ``detached``
        we hold no installation any more. Something called :func:`restore` —
        a mid-session ``disarm()``, or a NESTED pytest session that also
        enabled the tripwire, whose ``pytest_unconfigure`` restores the
        original and disarms the outer one along with itself.
    ``foreign-patch``
        we still hold an installation and the live entry is not ours: someone
        rebound the registry over us. The same condition :func:`restore`
        reports, reached before the report is written instead of after.

    Returns a string, like everything else here.
    """
    if not _installed:
        return "detached"
    if _installed["registry"].get(_installed["primitive"]) is _installed["wrapper"]:
        return "armed"
    return "foreign-patch"


def selfcheck() -> str:
    """Probe the armed hook **in both directions**, and return a status code.

    A positive-only self-check passes on a hook replaced by "record
    everything", and this repository has repeatedly found exactly that shape
    of vacuous control. So:

    ``not-invoked``
        a trace that must produce exactly one finding produced none. The
        wrapper is attached and blind — which is what a version bump actually
        produces, and what a naive ``hasattr`` misses.
    ``cries-wolf``
        a trace whose value **fits** produced a finding. The semantics moved;
        every "fire" this run would be noise.
    ``mis-attributed``
        the hook fired the right number of times and got the content wrong —
        the wrong value, the wrong dtype, the wrong file or the wrong line, or
        a narrowing whose independent recomputation disagrees with what was
        observed. A finding that points at the wrong line is worse than no
        finding, so it disables the tool rather than shipping.
    ``armed``
        all three behaved.

    Both directions reach the const-fold site — measured, one invocation each
    — and they differ in whether a *finding* results, not in whether the hook
    ran. That is what makes ``not-invoked`` and ``cries-wolf`` two different
    failures rather than two names for one, and it is why the negative
    direction also asserts ``invocations > 0``: a hook that stopped running
    altogether would otherwise pass the negative direction by producing
    nothing, exactly as a correct one does.

    The probe runs ``_probe.over`` / ``_probe.under`` rather than local
    lambdas so that the attributed frame is a real module — the same path a
    user's code takes, including the stack walk and the source quote.

    EACH RUN USES A FRESH INPUT SHAPE, and that is not tidiness. Measured on
    both tested series: jax's trace cache is process-wide and outlives
    disarm/rearm, so tracing ``_probe.over`` at the *same* avals reaches the
    const-fold site ``[1, 0, 0]`` times over three runs — once, and then never
    again. (``f86bafe`` recorded that sequence as "same shape three times, 0
    invocations each", which is wrong in its first element: the first trace
    does reach the site, which is the only reason a fixed-shape probe passed
    at all. Shapes 1, 2, 3 give ``[1, 1, 1]``, as that commit says.) A probe with a fixed shape therefore passes exactly
    once per process and reports ``not-invoked`` for ever after — which would
    make a second ``arm()`` in one process, and every test that arms twice,
    look like a broken hook. The shape is the cheapest thing in the cache key
    that is ours to vary.

    The recorder's counters are saved and restored: a self-check must not
    appear in the user's denominator.

    IT TAKES NO RECORDER, and that is a correction rather than a style
    choice. It used to take one, and a caller who passed anything other than
    the recorder :func:`install` closed over got ``not-invoked`` from a
    perfectly live hook — the probe read a counter nothing was writing to.
    The wrapper's recorder is the only one that can answer, so the wrapper's
    recorder is the one this reads.
    """
    recorder = _installed.get("recorder")
    if recorder is None:
        return "not-invoked"
    from stelling._jax_compat import jax as _jax  # public jax, via the boundary
    from stelling._jax_compat import jnp as _jnp

    from stelling._tripwire import _probe

    global _probe_seq
    _probe_seq += 1
    shape = (_probe_seq,)

    saved = recorder.as_payload()
    # Isolate the probe from any active gate: push our own stack entry so
    # probe fires go HERE and not into a gate that happens to be active.
    stack = _gate_fire_stack()
    stack.append(0)
    try:
        recorder.reset()
        try:
            _jax.make_jaxpr(_probe.over)(_jnp.zeros(shape, _jnp.int8))
        except Exception as exc:  # noqa: BLE001
            return f"unexpected:{type(exc).__name__}"
        if recorder.invocations == 0:
            return "not-invoked"
        if recorder.count != 1:
            return "not-invoked"

        found = next(iter(recorder.findings.values()))
        if (
            found.written != _probe.OVER
            or found.to_dtype != _probe.DTYPE
            or not found.agrees
            or os.path.abspath(found.file) != os.path.abspath(_probe.__file__)
            or found.line != _probe.OVER_LINE
        ):
            return "mis-attributed"

        recorder.reset()
        try:
            _jax.make_jaxpr(_probe.under)(_jnp.zeros(shape, _jnp.int8))
        except Exception as exc:  # noqa: BLE001
            return f"unexpected:{type(exc).__name__}"
        if recorder.invocations == 0:
            return "not-invoked"
        if recorder.count != 0:
            return "cries-wolf"
        return "armed"
    finally:
        stack.pop()
        recorder.reset()
        recorder.absorb(saved)


# ===========================================================================
# THE EAGER CONSTRUCTION SITE (Mode 2)
#
# A SECOND private-jax hook, in the one file allowed to have one. It is here
# and not in a new module because the exemption in
# `design/private-jax-boundary.md` is pinned to this exact path by two
# independent controls -- the `jax-import-hygiene` pre-commit hook's anchored
# filter and `tests/test_import_hygiene.py`'s repo-relative comparison -- and
# widening an exemption is a boundary change, not a convenience. What the
# exemption bought was "one file may name a private jax module", not "one
# private jax module may be named": a second hook inside the same file costs
# the boundary nothing, and a second file would cost it its shape.
#
# WHAT IT ATTACHES TO, AND WHY IT IS AN ATTRIBUTE AND NOT A REGISTRY. The
# const-fold tripwire above attaches to a registry ENTRY keyed on a primitive.
# There is no registry here: the narrowing happens in straight-line code inside
# `jax._src.lax.lax._convert_element_type`, at
#
#     if type(operand) is int and new_dtype != dtypes.float0:
#       arr = np.asarray(operand).astype(new_dtype)
#
# and, one branch down, at the `isinstance(operand, np.ndarray)` `.astype`.
# Measured on jax 0.11.0 and 0.10.2: every construction route that loses a
# written constant eagerly reaches this one function WITH THE WRITTEN VALUE
# INTACT, and every caller inside jax reaches it by MODULE ATTRIBUTE -- there
# are zero `from ... import _convert_element_type` in the installed tree -- so
# one attribute patch covers all of them at once.
#
# PUBLIC SURFACES ARE NOT SUFFICIENT, measured rather than assumed:
# `jnp.full is not jax.lax.full`, and patching both of those public names gets
# **0 hits** on `jnp.full_like` and on `jnp.stack`-of-`full`. There is no route
# to this door that does not name a private module, which is the same finding
# the const-fold hook rests on, one layer over.
# ===========================================================================

#: The private module the eager narrowing lives in, and the attribute on it.
#: Data rather than an import statement, for the same reason
#: :data:`PRIVATE_MODULE` is.
EAGER_MODULE = "jax." + "_src.lax.lax"
EAGER_ATTR = "_convert_element_type"

#: The parameters :func:`_make_eager_wrapper` reads, by name and position. The
#: wrapper forwards ``*args, **kwargs`` verbatim, so it does not depend on the
#: rest of the signature -- but it DOES depend on ``operand`` being first and
#: ``new_dtype`` second, and on both being passable positionally. That is the
#: drift this pins, and :func:`eager_signature_check` refuses to arm when it
#: moves. A hook that silently read the wrong argument would report a
#: truncation that did not happen, at a line that did not write it, which is
#: the one failure mode the report says costs more trust than a missing
#: finding.
EAGER_SIGNATURE = ("operand", "new_dtype")

#: sha1[:12] of the eager narrowing function's source, keyed on the exact jax
#: release, on exactly the discipline :data:`_KNOWN_HASHES` documents at
#: length: a MAP and not a set, recorded and never gated on, every row written
#: by someone who read the diff.
#:
#: THE SOURCE MOVES WHERE THE NARROWING DOES NOT, and this table is the
#: evidence for that claim rather than an assertion of it: the three hashes
#: below are three different strings, and the two lines that narrow are
#: character-identical across all three. That is exactly why arming is gated
#: on the BEHAVIOUR (:func:`eager_selfcheck` drives every route positively)
#: and never on the hash.
_KNOWN_EAGER_HASHES: dict[str, str] = {
    # Read and diffed, not copied in. 0.10.2 -> 0.11.0 is ONE hunk, five lines
    # added after the branch this hook exists for:
    #
    #     + elif (isinstance(operand, np.ndarray) and
    #     +       operand.dtype != dtypes.float0 and new_dtype != dtypes.float0):
    #     +   arr = operand.astype(new_dtype, copy=False)
    #     +   aval = core.ShapedArray(arr.shape, arr.dtype, weak_type=weak_type)
    #     +   operand = literals.TypedNdArray(arr, aval=aval)
    #
    # It ADDS a narrowing branch (a 0-d NumPy array now narrows here rather
    # than downstream at the bind) and leaves `if type(operand) is int: arr =
    # np.asarray(operand).astype(new_dtype)` character-identical. This hook
    # reads the OPERAND at function entry, before either branch, so both
    # releases are watched by the same wrapper and the added branch changes
    # nothing about what is seen -- driven on both, seven routes each.
    "0.10.2": "1390fde39809",
    "0.11.0": "17355ab7e4e1",
    # 0.11.0 -> 0.11.1: the function is BYTE-IDENTICAL. Read out of the
    # 0.11.1 wheel and hashed by the same slice this file takes at run time
    # (`inspect.getsource` and an `ast` extraction were checked against each
    # other on both installed series and agree). Note the contrast with
    # `_KNOWN_HASHES` directly above, where 0.11.0 and 0.11.1 DIFFER and
    # 0.10.2 and 0.11.0 agree: the two sites move on different releases,
    # which is the whole reason each carries its own map rather than sharing
    # one "the jax internals moved" flag.
    "0.11.1": "17355ab7e4e1",
}

#: What this process installed over :data:`EAGER_ATTR`, or empty. Separate
#: from :data:`_installed` because the two hooks arm, disarm and fail
#: independently: a user may want the eager detector and not the tripwire, and
#: a release may move one site and not the other.
_eager_installed: dict = {}


def _eager_module():
    """The module carrying the narrowing, or None. Never raises."""
    try:
        return importlib.import_module(EAGER_MODULE)
    except ImportError:
        return None


def _eager_original():
    """jax's own function, whether or not we are armed over it."""
    if _eager_installed:
        return _eager_installed["original"]
    module = _eager_module()
    return None if module is None else getattr(module, EAGER_ATTR, None)


def eager_locate() -> str:
    """Whether the eager narrowing site is where this expects it.

    ``no-module``
        jax is not installed.
    ``no-site-module``
        jax is installed and ``jax._src.lax.lax`` did not import. A release
        moved the module; refusing to guess is the designed behaviour.
    ``no-site``
        the module is there and carries no ``_convert_element_type``. The
        eager narrowing this watches may now happen somewhere else entirely.
    ``located``
        both are there.
    """
    if not _optional.available("jax"):
        return "no-module"
    module = _eager_module()
    if module is None:
        return "no-site-module"
    return "located" if getattr(module, EAGER_ATTR, None) is not None else "no-site"


def eager_signature_check() -> str:
    """``"ok"``, or a sentence naming exactly how the signature drifted.

    THIS IS THE HALF OF FAIL-CLOSED A PRESENCE CHECK MISSES. ``locate()``
    answers "is there a function called that", and a jax release that reordered
    the parameters would leave it answering yes while the wrapper read a
    ``sharding`` where it expected a dtype. The behaviour probe in
    :func:`eager_selfcheck` would very likely catch it too -- but "very likely"
    is not the standard for an instrument whose failure is a wrong line number
    in an alarm, and this check costs one :func:`inspect.signature` call at arm
    time.

    Only the first two parameters are pinned, and only their names, order and
    kind. Everything after them is forwarded verbatim, so a release that ADDS
    a parameter must not disable the tool -- that is the same
    additive-changes-are-free property :func:`_make_wrapper` argues for above.
    """
    original = _eager_original()
    if original is None:
        return "the eager narrowing site is not present"
    try:
        parameters = list(inspect.signature(original).parameters.values())
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        return f"the signature could not be read: {type(exc).__name__}: {exc}"
    if len(parameters) < len(EAGER_SIGNATURE):
        return (
            f"{EAGER_MODULE}.{EAGER_ATTR} now takes {len(parameters)} "
            f"parameter(s); this hook reads the first {len(EAGER_SIGNATURE)}"
        )
    for index, expected in enumerate(EAGER_SIGNATURE):
        parameter = parameters[index]
        if parameter.name != expected:
            return (
                f"parameter {index} of {EAGER_MODULE}.{EAGER_ATTR} is "
                f"{parameter.name!r} and this hook reads it as {expected!r}"
            )
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return (
                f"parameter {index} of {EAGER_MODULE}.{EAGER_ATTR} "
                f"({parameter.name!r}) is no longer passable positionally "
                f"({parameter.kind})"
            )
    return "ok"


def eager_site_hash() -> str | None:
    """sha1[:12] of the narrowing function's source, or None if unreadable.

    Recorded in the status; never gated on, for the reason :data:`_KNOWN_HASHES`
    gives at length.
    """
    original = _eager_original()
    if original is None:
        return None
    try:
        return hashlib.sha1(inspect.getsource(original).encode()).hexdigest()[:12]
    except Exception:  # noqa: BLE001 - a disclosure may not raise
        return None


def eager_site_name() -> str | None:
    """The narrowing function's ``__name__``. Recorded, not gated on."""
    return getattr(_eager_original(), "__name__", None)


def known_eager_hash() -> str | None:
    """The row :data:`_KNOWN_EAGER_HASHES` holds for the RUNNING release.

    ``None`` means *this jax has never been read*, which is a third state and
    not a synonym for "changed" -- :meth:`_tripwire.Status.hash_state` is the
    one place that four-way distinction is made, and this feeds it.
    """
    return _KNOWN_EAGER_HASHES.get(jax_version() or "")


def _eager_dtype_name(new_dtype) -> str | None:
    """``new_dtype`` as one of :data:`record.INT_DTYPES`'s names, or None.

    Asks the object for its own name and never converts it. Nothing here may
    import numpy: this module must import cleanly with jax absent, and jax
    absent means numpy may be absent too.
    """
    name = getattr(new_dtype, "name", None)
    if not isinstance(name, str):
        name = getattr(new_dtype, "__name__", None)
    if not isinstance(name, str):
        name = str(new_dtype)
    return name if name in record.INT_DTYPES else None


def _eager_scalar_int(operand):
    """The written integer, if this operand is one. None otherwise.

    THREE SPELLINGS REACH THE NARROWING WITH THE WRITTEN VALUE INTACT, measured
    on both series, and they take two different branches inside jax:

    * a Python ``int`` -- ``type(operand) is int``, the branch whose
      ``np.asarray(operand).astype(new_dtype)`` is the site this hook exists
      for. Every one of ``jnp.full``, ``jnp.full_like``, ``lax.full``,
      ``lax.full_like``, ``lax.convert_element_type`` and anything built on
      them arrives here.
    * a NumPy scalar (``np.int64(256)``) -- ``np.generic``, which reaches
      ``np.asarray(operand, dtype=new_dtype)``, raises ``OverflowError``
      INSIDE jax's own ``try``, is swallowed by design (jax's comment says so),
      and narrows downstream instead. ``SOUNDNESS.md`` names this spelling
      explicitly as one it could not close.
    * a 0-d NumPy array -- ``operand.astype(new_dtype, copy=False)``.

    **Scalars only, and jax objects never.** A non-scalar array reaching
    ``.astype`` is a user converting DATA, not writing a constant, and firing
    on it would make the detector unusable on any real program. A jax
    ``Array`` or ``Tracer`` is a value the program computes; its conversion is
    a ``convert_element_type`` the program performs at RUN time, which is the
    ``deferred`` bucket in ``GATE_COVERAGE`` and is not a transcription loss at
    all. The module test rather than an ``isinstance`` keeps this module free
    of numpy and free of jax alike.
    """
    if type(operand) is int:
        return operand
    if type(operand).__module__.split(".")[0] != "numpy":
        return None
    shape = getattr(operand, "shape", None)
    dtype = getattr(operand, "dtype", None)
    if shape != () or dtype is None:
        return None
    if _eager_dtype_name(dtype) is None:
        return None
    try:
        return int(operand)
    except Exception:  # noqa: BLE001 - a probe may not raise
        return None


#: Errors the eager wrapper's OWN bookkeeping raised and swallowed. A plain
#: module counter and NOT a key in :data:`_eager_installed`, which is where it
#: lived for one draft: writing a key into that dict makes it truthy, and
#: every reader there guards with ``if not _eager_installed`` and then indexes
#: ``["module"]``. An orphaned wrapper -- one still bound after
#: :func:`eager_restore` cleared the record -- would therefore have turned its
#: own swallowed error into a ``KeyError`` out of :func:`eager_is_armed`. The
#: count belongs to the process, not to an installation.
_eager_internal_errors = [0]


def _make_eager_wrapper(original):
    """Wrap the narrowing site. The observer is read per call, by design.

    ``*args, **kwargs`` FORWARDED VERBATIM. The wrapper never re-supplies a
    default and never reorders anything: whatever jax passed is what jax's own
    function receives, so a release that adds a parameter, changes a default,
    or starts passing an existing one by keyword costs this hook nothing. The
    two arguments it READS are pulled out by position-or-name, and
    :func:`eager_signature_check` refuses to arm if either has moved.

    THE OBSERVER IS LOOKED UP IN ``_eager_installed`` ON EVERY CALL rather than
    closed over. That is what lets :func:`eager_selfcheck` swap in a collector
    and drive the live wrapper -- the same object a user's code will meet --
    instead of probing a re-implementation of it, which is the shape of vacuous
    control this repository keeps having to withdraw.

    THE OBSERVER IS CALLED BEFORE ``original``, and the raise it may perform
    escapes through this frame untouched. Before, so that the truncation is
    reported instead of performed: jax's narrowing has already happened by the
    time ``original`` returns, and an alarm raised after it would be an alarm
    about a value that had already been destroyed in a jaxpr somebody might be
    holding. Untouched, because the whole point of a ``BaseException`` is that
    handlers written for other purposes do not get to interfere -- including
    this one's, which is why the ``except`` below re-raises it by type before
    the blanket clause can count it as an internal error.
    """

    # CAPTURED AT BUILD TIME, not imported inside the handler. The handler
    # runs on the one path where something has already gone wrong, and an
    # import there is one more thing that can fail while deciding whether the
    # thing that failed was the alarm.
    from stelling._tripwire.eager import EagerTruncationError

    def stelling_eager_truncation_probe(*args, **kwargs):
        try:
            operand = args[0] if args else kwargs.get("operand")
            new_dtype = args[1] if len(args) > 1 else kwargs.get("new_dtype")
            if new_dtype is not None:
                written = _eager_scalar_int(operand)
                if written is not None:
                    to_dtype = _eager_dtype_name(new_dtype)
                    if to_dtype is not None:
                        observer = _eager_installed.get("observer")
                        if observer is not None:
                            observer(written, to_dtype)
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (EagerTruncationError, KeyboardInterrupt, SystemExit)):
                raise
            # An instrument that breaks the program it measures is worse than
            # no instrument -- the same rule the const-fold wrapper follows,
            # with the one exception this hook exists to raise carved out
            # above by type rather than by hope.
            _eager_internal_errors[0] += 1
        return original(*args, **kwargs)

    return stelling_eager_truncation_probe


def eager_install(observer) -> str:
    """Patch the narrowing site. ``installed``, ``already-armed``, or a code.

    Never raises and never leaks a jax object: ``observer`` is a plain callable
    of ``(int, str)`` and everything returned here is a string.
    """
    code = eager_locate()
    if code != "located":
        return code
    if _eager_installed:
        _eager_installed["observer"] = observer
        return "already-armed"
    try:
        module = _eager_module()
        original = getattr(module, EAGER_ATTR)
        wrapper = _make_eager_wrapper(original)
        _eager_installed.update(
            {"module": module, "original": original,
             "wrapper": wrapper, "observer": observer}
        )
        setattr(module, EAGER_ATTR, wrapper)
    except Exception as exc:  # noqa: BLE001
        _eager_installed.clear()
        return f"unexpected:{type(exc).__name__}"
    return "installed"


def eager_restore() -> str:
    """Put jax's own function back, **by identity**.

    ``not-armed``, ``foreign-patch`` (someone rebound the attribute over us --
    say so rather than clobbering whatever they put there), or ``restored``.

    A PENDING :func:`eager_detach` IS FIXED UP FIRST, exactly as in
    :func:`restore`, and this is not tidying -- it was MEASURED here before it
    was written. ``eager_detach`` saves whatever the attribute held and
    ``eager_reattach`` puts it back, and what it held may be THIS WRAPPER.
    Retiring the wrapper makes that saved entry stale, and the sequence
    ``eager_detach("rebind")`` -> ``disarm_eager()`` (which returns
    ``foreign-patch`` and clears our record) -> ``eager_reattach()`` then
    binds an ORPHANED wrapper as the live function: no ``_eager_installed``
    record owns it, so its observer lookup finds nothing and it silently does
    not watch, while ``eager_signature_check`` reads ITS signature
    (``*args, **kwargs``) as jax's and reports ``signature-drift`` for ever
    after. Driven: four later tests in ``tests/test_tripwire_eager.py``
    skipped with ``signature-drift`` against a jax whose function had not
    moved. ``eager_restore`` is the operation that invalidates the saved
    entry, so it is the operation that corrects it.
    """
    if not _eager_installed:
        return "not-armed"
    module = _eager_installed["module"]
    wrapper = _eager_installed["wrapper"]
    original = _eager_installed["original"]
    if _eager_detached and _eager_detached.get("entry") is wrapper:
        _eager_detached["entry"] = original
    current = getattr(module, EAGER_ATTR, None)
    if current is not wrapper:
        _eager_installed.clear()
        return "foreign-patch"
    setattr(module, EAGER_ATTR, original)
    _eager_installed.clear()
    return "restored"


def eager_is_armed() -> bool:
    """Whether the wrapper this process installed is still the live attribute."""
    if not _eager_installed:
        return False
    module = _eager_installed["module"]
    return getattr(module, EAGER_ATTR, None) is _eager_installed["wrapper"]


def eager_live_check() -> str:
    """``armed``, ``detached`` or ``foreign-patch``. Never raises.

    Same three states, and the same reason for keeping the two "no"s apart, as
    :func:`live_check`: ``detached`` means we hold no installation any more
    and ``foreign-patch`` means we hold one and are not the live entry. They
    are different failures and lead a reader to different places.
    """
    if not _eager_installed:
        return "detached"
    module = _eager_installed["module"]
    if getattr(module, EAGER_ATTR, None) is _eager_installed["wrapper"]:
        return "armed"
    return "foreign-patch"


def eager_detach(mode: str) -> str:
    """Break the eager hook the way a jax release would. Test support.

    Shipped rather than written in ``tests/`` for the reason :func:`detach` is:
    rule 2 bans naming the private jax module in ``tests/`` with no exemption,
    so a test that reached in to break this would have to name what only this
    file may name. Asserting the fail-closed contract instead of driving it is
    the shape of control this repository has been burned by.

    ``mode="bypass"``
        put jax's own function back while this process still believes it is
        armed. The hook is attached and never invoked -- what a version bump
        actually produces, and what a presence check misses.
    ``mode="rebind"``
        bind a THIRD function over the top, which is the ``foreign-patch``
        displacement: neither ours nor jax's is live, and nothing about our
        own bookkeeping has changed.
    ``mode="signature"``
        bind a function whose FIRST TWO PARAMETERS have moved, which is the
        drift :func:`eager_signature_check` exists to refuse. It delegates, so
        jax still works and only the signature has changed -- the point being
        that a hook reading argument 0 as the operand would go on "working"
        and reporting the wrong thing.
    """
    module = _eager_module()
    if module is None:
        return "no-site-module"
    if _eager_detached:
        return "already-detached"
    current = getattr(module, EAGER_ATTR, None)
    _eager_detached.update({"module": module, "entry": current})
    if mode == "bypass":
        setattr(module, EAGER_ATTR, _eager_installed.get("original", current))
        return "detached"
    if mode == "rebind":
        original = _eager_installed.get("original", current)

        def foreign(*args, **kwargs):
            return original(*args, **kwargs)

        setattr(module, EAGER_ATTR, foreign)
        return "detached"
    if mode == "signature":
        original = _eager_installed.get("original", current)

        def drifted(sharding=None, operand=None, new_dtype=None, *args, **kwargs):
            return original(operand, new_dtype, *args, **kwargs)

        setattr(module, EAGER_ATTR, drifted)
        return "detached"
    _eager_detached.clear()
    return f"unknown-mode:{mode}"


def eager_reattach() -> str:
    """Undo :func:`eager_detach`. ``not-detached`` if there is nothing to undo."""
    if not _eager_detached:
        return "not-detached"
    setattr(_eager_detached["module"], EAGER_ATTR, _eager_detached["entry"])
    _eager_detached.clear()
    return "reattached"


_eager_detached: dict = {}

#: The value the eager self-check writes, the dtype it writes it into, and the
#: value that must survive. 256 into ``int8`` is one step over the edge, and 3
#: is comfortably inside it -- the same pair, and for the same reason, as
#: :mod:`_probe`'s.
EAGER_OVER = 256
EAGER_UNDER = 3
EAGER_DTYPE = "int8"


def _eager_routes():
    """``(name, build)`` for every construction route this detector claims.

    Each ``build(value)`` writes ``value`` into an ``int8`` array by one
    spelling and returns the array. THE LIST IS THE CLAIM: :func:`eager_selfcheck`
    drives every entry positively and refuses to arm if any one of them stops
    reaching the hook, so a route added here without a hook that sees it fails
    closed rather than quietly widening what the tool says it covers.

    The last entry is a different jax BRANCH and not just a different spelling
    -- ``np.int64(256)`` reaches ``np.asarray(operand, dtype=new_dtype)``
    rather than the ``type(operand) is int`` ``.astype`` -- so it is the one
    that would notice jax fixing one branch and not the other.
    """
    from stelling._jax_compat import jax as _jax
    from stelling._jax_compat import jnp as _jnp

    lax = _jax.lax
    dt = getattr(_jnp, EAGER_DTYPE)

    def like(value):
        return _jnp.zeros((3,), dt)

    return (
        ("jnp.full", lambda v: _jnp.full((), v, dt)),
        ("jnp.full_like", lambda v: _jnp.full_like(like(v), v)),
        ("lax.full", lambda v: lax.full((), v, dt)),
        ("lax.full_like", lambda v: lax.full_like(like(v), v)),
        ("lax.convert_element_type", lambda v: lax.convert_element_type(v, dt)),
        ("jnp.stack of jnp.full", lambda v: _jnp.stack([like(v), _jnp.full((3,), v, dt)])),
        ("numpy scalar into jnp.full", lambda v: _jnp.full((), _np_scalar(v), dt)),
    )


def _np_scalar(value):
    """``np.int64(value)``, reached through jax's own numpy rather than ours.

    This module may not ``import numpy`` at module scope for the same reason it
    may not import jax: it has to import cleanly in a bare interpreter. Inside
    the self-check jax is already imported, and jax cannot exist without numpy.
    """
    numpy = importlib.import_module("numpy")
    return numpy.int64(value)


def eager_selfcheck() -> str:
    """Probe the armed hook over **every route it claims**, both directions.

    ``route-blind:<name>``
        a route that must reach the hook did not. This is the failure the whole
        design turns on: the attribute is still there, the wrapper is still
        installed, and jax has stopped routing one construction spelling
        through it. It is silent, it is what a jax release actually produces,
        and a presence check cannot see it. **It refuses to arm.** Keeping five
        routes and losing one quietly is not a trade this tool gets to make on
        a user's behalf -- the whole value of Mode 2 is that a program either
        cannot contain an undeclared truncation or the tool says it is not
        watching.
    ``cries-wolf``
        an IN-RANGE value was reported as out of range, so every alarm this run
        would be noise.
    ``mis-attributed``
        the hook reported a truncation and the array jax actually built does
        not hold the value this tool's own arithmetic predicts. A finding that
        describes something other than what happened costs more trust than a
        missing one.
    ``not-invoked``
        the wrapper is attached and saw nothing at all.
    ``armed``
        every route fired positively, none fired negatively, and the observed
        results agree with the independent recomputation.

    The observer is swapped for a collector for the duration, so the live
    wrapper -- the same object a user's code will meet -- is what is driven.
    The user's counters are untouched: nothing here reaches
    :func:`eager.observe`, so a self-check can never appear in a denominator.
    """
    if not _eager_installed:
        return "not-invoked"
    saved = _eager_installed.get("observer")
    seen: list[tuple[int, str]] = []
    _eager_installed["observer"] = lambda written, to_dtype: seen.append(
        (written, to_dtype)
    )
    try:
        try:
            routes = _eager_routes()
        except Exception as exc:  # noqa: BLE001
            return f"unexpected:{type(exc).__name__}"
        expected = record.narrow(EAGER_OVER, EAGER_DTYPE)
        for name, build in routes:
            del seen[:]
            try:
                built = build(EAGER_OVER)
            except Exception as exc:  # noqa: BLE001
                return f"unexpected:{type(exc).__name__}"
            if not any(
                written == EAGER_OVER and to == EAGER_DTYPE for written, to in seen
            ):
                return f"route-blind:{name}"
            try:
                got = int(built.ravel()[0]) if built.ndim else int(built)
            except Exception:  # noqa: BLE001
                return "mis-attributed"
            if got != expected:
                return "mis-attributed"

        del seen[:]
        for name, build in routes:
            try:
                build(EAGER_UNDER)
            except Exception as exc:  # noqa: BLE001
                return f"unexpected:{type(exc).__name__}"
        if not seen:
            return "not-invoked"
        if any(not record.in_range(written, to) for written, to in seen):
            return "cries-wolf"
        return "armed"
    finally:
        _eager_installed["observer"] = saved


def displacement_check() -> tuple[tuple[str, str], ...]:
    """Every hook this process installed, and whether it is STILL the live one.

    ONE INSTRUMENT FOR TWO DISPLACEMENTS, and the reason it is one and not two
    is a finding rather than a preference. B15's audit established that
    ``live_check() == "foreign-patch"`` is a FOURTH way the trace gate's watch
    goes partial and that the gate does not consult it: rebinding the registry
    entry after arming leaves the recorder's identity unchanged and
    ``fires_count()`` unchanged, so the gate's two existing partiality tests
    both pass, the fire counter stays at zero because our wrapper is no longer
    called, and ``check()`` returns **VERIFIED on a watched route**. Building
    the eager hook meant building a displacement instrument anyway; two of them
    -- one per hook, each consulted by a different caller -- is two chances to
    teach one caller about one hook and forget the other.

    Returns ``()`` when nothing is armed, which is not the same as "clear":
    a caller asks about the hooks it armed, and the empty tuple says none.
    ``(name, state)`` pairs, states from the same vocabulary
    :func:`live_check` uses.
    """
    states: list[tuple[str, str]] = []
    if _installed:
        states.append(("const-fold", live_check()))
    if _eager_installed:
        states.append(("eager", eager_live_check()))
    return tuple(states)


def displaced() -> tuple[str, ...]:
    """The names of the armed hooks that are NOT live. Empty when all are.

    The predicate form of :func:`displacement_check`, because that is what
    every caller actually wants and computing it in three places is three
    chances to spell ``!= "armed"`` as ``== "detached"``.
    """
    return tuple(
        name for name, state in displacement_check() if state != "armed"
    )
