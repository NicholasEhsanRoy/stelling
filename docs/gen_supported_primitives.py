# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Generate ``docs/supported-primitives.md`` from the live registries.

The page is DERIVED, never transcribed: every membership, count, and
citation below is computed by importing the registries and scanning the
source files at generation time. Reasons quoted on the page are verified
verbatim against the code — if a quoted comment/docstring no longer
exists in the file, generation fails loudly rather than citing stale
text. ``tests/test_supported_primitives_doc.py`` regenerates the page
and fails if the committed copy differs from the live registries.

Registries consumed (located by scan, cited with live line numbers):

- ``stelling.propagate.TRANSFERS`` — real-mode interval transfers, tiered
- ``stelling.propagate.IEEE_TRANSFERS`` — ieee-mode transfers, tiered
- ``stelling.propagate._INT_COMPUTING`` / ``_INT_NON_COMPUTING`` /
  ``_INT_NON_COMPUTING_EXEMPT`` — the transfer-side integer-semantics census
- ``stelling.propagate._ASSUME_CMPS`` — assume-narrowing comparisons
- ``stelling.obligation._SUPPORTED`` — the SMT emission set
- ``stelling.obligation._INT_OVERFLOW_EMITTED`` / ``_INT_SAFE_EMITTED`` /
  ``_INT_SAFE_EMITTED_REASONS`` — the emission-side integer-semantics census
- ``stelling.obligation._REPLAY_SUPPORTED`` — the exact-rational replay surface
- ``stelling.obligation._SCALAR_STRUCT_FMT`` — the scalar literal decoder (dtype-keyed)
- ``stelling.coverage.DEFAULT_TRANSPARENT`` — transparent call wrappers

Usage::

    python docs/gen_supported_primitives.py            # rewrite the page
    python docs/gen_supported_primitives.py --check    # exit 1 on drift
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_DOCS = Path(__file__).resolve().parent
_ROOT = _DOCS.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stelling import coverage as _cov  # noqa: E402
from stelling import obligation as _ob  # noqa: E402
from stelling import propagate as _pr  # noqa: E402

PAGE = _DOCS / "supported-primitives.md"

_PROPAGATE = "src/stelling/propagate.py"
_OBLIGATION = "src/stelling/obligation.py"
_COVERAGE = "src/stelling/coverage.py"

_GENERATOR = "docs/gen_supported_primitives.py"
_TEST = "tests/test_supported_primitives_doc.py"


# -- source scanning ----------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(text.split())


def _lines(rel: str) -> list[str]:
    return (_ROOT / rel).read_text(encoding="utf-8").splitlines()


def _def_line(rel: str, name: str) -> int:
    """1-based line where ``name`` is assigned at module top level."""
    pat = re.compile(rf"^{re.escape(name)}\s*[:=]")
    for i, line in enumerate(_lines(rel), start=1):
        if pat.match(line):
            return i
    raise LookupError(f"definition of {name!r} not found in {rel}")


# Everything a wrapped string literal puts at a continuation: the closing
# quote of one fragment, then an optional prefix and the opening quote of
# the next. Dropping the quotes but keeping the prefix is the trap — see
# ``scrub`` below.
_LITERAL_PREFIX = re.compile(r"\b[fFrRbBuU]{1,2}(?=[\"'])")


def _quote_line(rel: str, quote: str, *, window: int = 40) -> int:
    """1-based line where the (whitespace-normalized) ``quote`` starts.

    The quote must appear verbatim in the file, modulo line wrapping,
    leading comment markers, string-literal syntax, and whitespace —
    otherwise generation fails: the page never cites text the code no
    longer carries.
    """
    def scrub(text: str) -> str:
        # comment markers, string-literal prefixes, and quote characters
        # are wrapping syntax, not content — scrub all three before
        # matching. The prefix has to go with its quote: a message
        # wrapped between ``f"... emission "`` and ``f"set: ..."``
        # otherwise reconstructs as "... emission f set: ...", and a
        # sentence the code still carries verbatim stops matching for no
        # reason but where the line happened to break.
        return _norm(
            _LITERAL_PREFIX.sub(" ", text)
            .replace("#", " ").replace('"', " ").replace("'", " ")
        )

    raw = _lines(rel)
    norm = [scrub(line) for line in raw]
    target = scrub(quote)
    n = len(raw)
    for i in range(n):
        acc = norm[i]
        j = i
        while target not in acc and j + 1 < min(i + window, n):
            j += 1
            acc = f"{acc} {norm[j]}"
        if target in acc:
            start = i
            while start < j and target in " ".join(norm[start + 1 : j + 1]):
                start += 1
            return start + 1
    raise LookupError(f"quote not found in {rel}: {quote[:70]!r}...")


def _q(rel: str, quote: str) -> str:
    """Render a code-recorded reason: verbatim quote plus file:line cite."""
    line = _quote_line(rel, quote)
    return f'"{_norm(quote)}" ({rel}:{line})'


def _asserted_tiers(text: str) -> set[str]:
    """Tier names the text itself asserts (\"tier <name>\" phrases).

    Feeds the generation-time consistency check on the tier-difference
    rows: a hand-attached quote is scan-verified for existence, but its
    CONTENT can still contradict the live tier it is cited to explain —
    exactly what happened with sqrt's ieee registry comment (\"tier sound
    not sound-libm\" beside a live ``TIER_EXACT``). Longest tier name
    first so ``sound-libm`` is never read as ``sound``.
    """
    names = sorted(
        (_pr.TIER_EXACT, _pr.TIER_SOUND, _pr.TIER_SOUND_LIBM),
        key=len, reverse=True,
    )
    pat = re.compile(
        r"tier\s+(" + "|".join(re.escape(n) for n in names) + r")\b"
    )
    return set(pat.findall(_norm(text)))


# -- table helpers ------------------------------------------------------------

def _cell(value: str) -> str:
    return value.replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return out


def _prims(names) -> str:
    return ", ".join(f"`{p}`" for p in sorted(names)) if names else "(none)"


# -- the page -----------------------------------------------------------------

def generate() -> str:
    transfers = _pr.TRANSFERS
    ieee = _pr.IEEE_TRANSFERS
    supported = set(_ob._SUPPORTED)
    replay = set(_ob._REPLAY_SUPPORTED)
    overflow = set(_ob._INT_OVERFLOW_EMITTED)
    int_safe = set(_ob._INT_SAFE_EMITTED)
    computing = set(_pr._INT_COMPUTING)
    non_computing = set(_pr._INT_NON_COMPUTING)
    transparent = set(_cov.DEFAULT_TRANSPARENT)
    assume_cmps = set(_pr._ASSUME_CMPS)
    compare = set(_ob._COMPARE)

    universe = sorted(set(transfers) | set(ieee) | supported | replay)

    regs = [
        ("stelling.propagate.TRANSFERS", _PROPAGATE,
         _def_line(_PROPAGATE, "TRANSFERS"), len(transfers),
         "real-mode interval transfer registry (`semantics=\"real\"`); each "
         "entry carries an assumption tier"),
        ("stelling.propagate.IEEE_TRANSFERS", _PROPAGATE,
         _def_line(_PROPAGATE, "IEEE_TRANSFERS"), len(ieee),
         "ieee-mode interval transfer registry (`semantics=\"ieee\"`); each "
         "entry carries an assumption tier"),
        ("stelling.propagate._INT_COMPUTING", _PROPAGATE,
         _def_line(_PROPAGATE, "_INT_COMPUTING"), len(computing),
         "transfer-side integer-semantics census: transfers that can compute "
         "a new numeric value (they carry the overflow-reachability guard)"),
        ("stelling.propagate._INT_NON_COMPUTING", _PROPAGATE,
         _def_line(_PROPAGATE, "_INT_NON_COMPUTING"), len(non_computing),
         "transfer-side integer-semantics census: transfers recorded as "
         "computing no new value"),
        ("stelling.propagate._INT_NON_COMPUTING_EXEMPT", _PROPAGATE,
         _def_line(_PROPAGATE, "_INT_NON_COMPUTING_EXEMPT"),
         len(_pr._INT_NON_COMPUTING_EXEMPT),
         "per-primitive written soundness reasons for the non-computing "
         "classification (reproduced in the appendix below)"),
        ("stelling.propagate._ASSUME_CMPS", _PROPAGATE,
         _def_line(_PROPAGATE, "_ASSUME_CMPS"), len(assume_cmps),
         "the comparisons a point-bounded `stelling_assume` can narrow "
         "through"),
        ("stelling.obligation._SUPPORTED", _OBLIGATION,
         _def_line(_OBLIGATION, "_SUPPORTED"), len(supported),
         "the SMT emission set: primitives an obligation slice may contain "
         "and emit"),
        ("stelling.obligation._INT_OVERFLOW_EMITTED", _OBLIGATION,
         _def_line(_OBLIGATION, "_INT_OVERFLOW_EMITTED"), len(overflow),
         "emission-side integer-semantics census: emitted primitives that "
         "compute a new numeric value (integer dtypes decline)"),
        ("stelling.obligation._INT_SAFE_EMITTED", _OBLIGATION,
         _def_line(_OBLIGATION, "_INT_SAFE_EMITTED"), len(int_safe),
         "emission-side integer-semantics census: emitted primitives "
         "recorded int-safe"),
        ("stelling.obligation._INT_SAFE_EMITTED_REASONS", _OBLIGATION,
         _def_line(_OBLIGATION, "_INT_SAFE_EMITTED_REASONS"),
         len(_ob._INT_SAFE_EMITTED_REASONS),
         "per-primitive written soundness reasons for the int-safe "
         "classification (reproduced in the appendix below)"),
        ("stelling.obligation._REPLAY_SUPPORTED", _OBLIGATION,
         _def_line(_OBLIGATION, "_REPLAY_SUPPORTED"), len(replay),
         "the exact-rational replay surface: primitives the solver-free "
         "witness replay can evaluate"),
        ("stelling.obligation._SCALAR_STRUCT_FMT", _OBLIGATION,
         _def_line(_OBLIGATION, "_SCALAR_STRUCT_FMT"),
         len(_ob._SCALAR_STRUCT_FMT),
         "the scalar literal decoder — keyed by numpy dtype code, not by "
         "primitive"),
        ("stelling.coverage.DEFAULT_TRANSPARENT", _COVERAGE,
         _def_line(_COVERAGE, "DEFAULT_TRANSPARENT"), len(transparent),
         "call wrappers descended into (sub-jaxpr walked) instead of "
         "transferred"),
    ]

    out: list[str] = []
    w = out.append

    # REUSE-IgnoreStart — these strings are emitted into the generated
    # page's own header; they are not a license declaration for this file
    # (this file's declaration is at the top).
    w("<!--")
    w("SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy")
    w("SPDX-License-Identifier: Apache-2.0")
    w("-->")
    # REUSE-IgnoreEnd
    w("")
    w("# Supported primitives")
    w("")
    w("**GENERATED FILE — do not edit by hand.** This page is emitted by")
    w(f"`{_GENERATOR}`, which imports stelling's live capability registries")
    w("and derives every membership, tier, count, and citation below from")
    w("them. The registries consumed, with their definition sites at")
    w("generation time:")
    w("")
    for name, rel, line, size, role in regs:
        w(f"- `{name}` (`{rel}:{line}`, {size} entries) — {role}")
    w("")
    w(f"Regenerate with `python {_GENERATOR}`. The drift gate")
    w(f"`{_TEST}` regenerates this page and fails")
    w("whenever the committed copy differs from the live registries.")
    w("")

    # -- per-primitive membership --------------------------------------------
    w("## Per-primitive membership")
    w("")
    w("One row per primitive named by any registry above (transparent call")
    w("wrappers are listed separately below). `—` means the primitive is not")
    w("a member of that registry.")
    w("")
    rows = []
    for p in universe:
        rows.append([
            f"`{p}`",
            transfers[p][1] if p in transfers else "—",
            ieee[p][1] if p in ieee else "—",
            ("computing" if p in computing
             else "non-computing" if p in non_computing else "—"),
            "emitted" if p in supported else "—",
            ("overflow-guarded" if p in overflow
             else "int-safe" if p in int_safe else "—"),
            "replayed" if p in replay else "—",
            "narrows" if p in assume_cmps else "—",
        ])
    out.extend(_table(
        ["primitive", "transfer (real)", "transfer (ieee)",
         "transfer int census", "SMT emission", "emission int census",
         "replay", "assume-narrowing"],
        rows,
    ))
    w("")

    # -- transparent wrappers ------------------------------------------------
    w("## Transparent call primitives")
    w("")
    w(f"`stelling.coverage.DEFAULT_TRANSPARENT` "
      f"({len(transparent)} members): {_prims(transparent)}.")
    w("")
    w("Recorded role: " + _q(_COVERAGE,
      "The wrapper primitives whose correct transfer is "
      "descend-into-sub-jaxpr, per design/transparent-primitives.md "
      "(verified on jax 0.10.2).") + ".")
    overlap = transparent & set(universe)
    if overlap:
        w("")
        w(f"Members that also appear in the registries above: "
          f"{_prims(overlap)}.")
    else:
        w("")
        w("No member of this set appears in any of the registries above.")
    w("")

    # -- counts --------------------------------------------------------------
    def tier_counts(reg):
        c: dict[str, int] = {}
        for _, tier in reg.values():
            c[tier] = c.get(tier, 0) + 1
        return ", ".join(f"{c.get(t, 0)} `{t}`" for t in
                         (_pr.TIER_EXACT, _pr.TIER_SOUND, _pr.TIER_SOUND_LIBM))

    w("## Counts")
    w("")
    w("All counts computed from the live registries at generation time.")
    w("")
    w(f"- {len(universe)} primitives appear in at least one registry "
      "(transparent call wrappers counted separately).")
    w(f"- {len(transfers)} primitives have a real-mode transfer: "
      f"{tier_counts(transfers)}.")
    w(f"- {len(ieee)} primitives have an ieee-mode transfer entry: "
      f"{tier_counts(ieee)}.")
    w(f"- Transfer-side integer census: {len(computing)} computing, "
      f"{len(non_computing)} non-computing, "
      f"{len(_pr._INT_NON_COMPUTING_EXEMPT)} written exemption reasons.")
    w(f"- {len(supported)} primitives are in the SMT emission set.")
    w(f"- Emission-side integer census: {len(overflow)} overflow-guarded, "
      f"{len(int_safe)} int-safe, "
      f"{len(_ob._INT_SAFE_EMITTED_REASONS)} written int-safe reasons.")
    w(f"- {len(replay)} primitives are on the exact-rational replay surface.")
    w(f"- {len(assume_cmps)} comparisons can narrow through a constraining "
      "assume.")
    w(f"- {len(transparent)} call wrappers are transparent.")
    w(f"- The scalar literal decoder covers "
      f"{len(_ob._SCALAR_STRUCT_FMT)} dtype codes (it is keyed by dtype, "
      "not by primitive).")
    w("")

    # -- where the sets differ -----------------------------------------------
    w("## Where the sets differ")
    w("")
    w("Every difference below is computed from the live registries. Reasons")
    w("are quoted verbatim from code comments/docstrings, with file:line;")
    w("where the code records no reason for a difference, the entry says")
    w("\"no recorded reason\".")
    w("")

    # real vs ieee membership
    w("### Real vs ieee transfer registry membership")
    w("")
    only_real = sorted(set(transfers) - set(ieee))
    only_ieee = sorted(set(ieee) - set(transfers))
    if not only_real and not only_ieee:
        w(f"The two transfer registries register exactly the same "
          f"{len(transfers)} primitives. This is enforced by an import-time")
        w("check whose recorded reason is: " + _q(_PROPAGATE,
          "the census must stay total: a registered transfer with no ieee "
          "census entry would be silent reuse, the exact thing rule 6 "
          "forbids") + ".")
    else:
        w(f"Real-only: {_prims(only_real)}. Ieee-only: {_prims(only_ieee)}.")
    w("")

    # tier differences
    w("### Real vs ieee assumption tiers")
    w("")
    flips = [p for p in sorted(set(transfers) & set(ieee))
             if transfers[p][1] != ieee[p][1]]
    w(f"{len(flips)} primitives carry different tiers in the two registries;")
    w(f"{len(set(transfers) & set(ieee)) - len(flips)} carry the same tier "
      "in both.")
    w("")
    w("The module docstring's recorded rationale for the ieee endpoint")
    w("arithmetic is: " + _q(_PROPAGATE,
      "endpoint arithmetic for the monotone core is native binary64 with NO "
      "outward rounding (the float value itself is computable — "
      ":data:`stelling.interval.IEEE_ENDPOINT_ASSUMPTION`)") + ".")
    w("")
    w("Per-primitive recorded reasons at the registry entries. Each")
    w("attached reason is checked at generation time for consistency with")
    w("the live ieee tier it is cited beside: a reason whose text asserts")
    w("a contradicting tier fails generation unless the row is presented")
    w("as a recorded discrepancy, and a discrepancy row whose recorded")
    w("text no longer contradicts the live tier fails generation too.")
    w("")
    # (file, verbatim quote, discrepancy-mark). The mark means: the quote
    # CONTRADICTS the live tier and is presented as a recorded
    # discrepancy, never as the explanation of the live tier.
    core_reason = (_PROPAGATE,
        "(ii) ieee variants: the monotone arithmetic core — native binary64 "
        "endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0 "
        "convention (iv._prod inside iv.mul) is NOT reused.", False)
    tier_reasons: dict[str, tuple[str, str, bool]] = {
        "add": core_reason,
        "sub": core_reason,
        "add_any": core_reason,
        "mul": core_reason,
        "div": core_reason,
        # marked discrepant: the comment's "tier sound" contradicts the
        # live TIER_EXACT — a stale comment on main, not this page's to fix
        "sqrt": (_PROPAGATE,
            "(ii) native binary64, CORRECTLY rounded — the float root "
            "bracketed exactly (no outward bump, tier sound not "
            "sound-libm); a negative arg is NaN routed to the flag, a "
            "maybe-NaN operand poisons the result", True),
        "reduce_sum": (_PROPAGATE,
            "(ii) censused DOWN to the association-free cases: <=2 "
            "contributors are exact (0 or 1 addition; IEEE add is "
            "commutative), >=3 declines — float addition is not "
            "associative and the jaxpr fixes no order", False),
        "integer_pow": (_PROPAGATE,
            "(ii) censused down the same way: y in {0, 1} perform NO "
            "arithmetic and are exact (y=0 is measured 1.0 even at NaN, so "
            "it CLEARS the flag); every other exponent declines — no fixed "
            "multiply schedule", False),
        "square": (_PROPAGATE,
            "`square` under ieee — DECLINES, and the reason is not schedule "
            "ambiguity.", False),
        "scatter-add": (_PROPAGATE,
            "Category (iii), whole-primitive: the censused ieee REFUSAL for "
            "scatter-add — the honest floor, chosen over an "
            "order-independent enclosure.", False),
        "dot_general": (_PROPAGATE,
            "dot_general under ieee: a whole-primitive censused REFUSAL.",
            False),
    }
    stale = sorted(set(tier_reasons) - set(flips))
    if stale:
        # an attachment for a primitive whose tiers no longer differ (or
        # that left the registries) is a stale claim; refuse rather than
        # silently skip it — a live-tier change must not bypass the
        # consistency check by dissolving the row it applied to
        raise LookupError(
            f"tier reasons attached to primitives whose real/ieee tiers no "
            f"longer differ: {stale} — remove or update the attachments"
        )
    for p in flips:
        head = f"- `{p}` — real `{transfers[p][1]}`, ieee `{ieee[p][1]}`: "
        entry = tier_reasons.get(p)
        if entry is None:
            w(head + "no recorded reason")
            continue
        rel, quote, discrepant = entry
        live = ieee[p][1]
        contradicting = sorted(_asserted_tiers(quote) - {live})
        if contradicting and not discrepant:
            raise LookupError(
                f"tier reason attached to {p!r} asserts tier(s) "
                f"{contradicting} but the live ieee tier is {live!r} — fix "
                f"the attachment, or mark the row as a recorded discrepancy"
            )
        if discrepant and not contradicting:
            raise LookupError(
                f"{p!r} is marked as a tier discrepancy but its recorded "
                f"text no longer contradicts the live ieee tier {live!r} — "
                f"unmark the row"
            )
        if discrepant:
            w(head + "RECORDED DISCREPANCY — the registry comment reads "
              + _q(rel, quote) + ", which asserts tier "
              + ", ".join(f"`{t}`" for t in contradicting)
              + f", contradicting the live tier `{live}` carried by the "
              "`IEEE_TRANSFERS` entry it annotates. The comment is stale "
              "on main, flagged for correction; it is quoted here as "
              "recorded text, not as the reason for the live tier.")
        else:
            w(head + _q(rel, quote))
    w("")

    # emission vs transfer
    w("### Emission set vs transfer registries")
    w("")
    em_not_tr = sorted(supported - set(transfers))
    tr_not_em = sorted(set(transfers) - supported)
    w(f"In the emission set but in neither transfer registry "
      f"({len(em_not_tr)}): {_prims(em_not_tr)}.")
    w("")
    assume_reasons = {
        "stelling_assume":
            _q(_OBLIGATION,
               "stelling_assume's *constraint* is inert (dropped, disclosed "
               "by the propagation notes) and is deliberately NOT emitted — "
               "only its data flow passes through, exactly as in "
               "propagation.")
            + "; on the propagation side it is handled by the walk itself "
            "rather than through the transfer registry: "
            + _q(_PROPAGATE,
                 "value semantics: the identity on the predicate — the "
                 "assume's output passes its input through unchanged in "
                 "BOTH modes"),
    }
    for p in em_not_tr:
        w(f"- `{p}` — "
          f"{assume_reasons.get(p, 'no recorded reason for the absence of a transfer')}")
    w("")
    w(f"In the transfer registries but not in the emission set "
      f"({len(tr_not_em)}): {_prims(tr_not_em)}.")
    w("")
    w("An unsupported primitive in a slice declines with the message "
      + _q(_OBLIGATION,
           "primitive {prim!r} is outside the supported emission set")
      + ". The module docstring's recorded decline classes are: "
      + _q(_OBLIGATION,
           "Everything else — over-budget slices, transcendentals, unknown "
           "primitives, possibly-zero divisor elements, non-float input "
           "declarations, obligations that cannot be mapped one-to-one onto "
           "top-level asserts — **declines**, with the primitive and form "
           "(and, for the budget, the count and the budget) quoted, and the "
           "obligation stays UNKNOWN.")
      + " Per-primitive recorded reasons:")
    w("")
    slice_endpoint_reason = _q(_OBLIGATION,
        "extracts the *expression slice* — the ir equations from the "
        "``stelling_any`` declarations and constants to the "
        "``stelling_assert`` operand")
    tr_not_em_reasons = {
        "stelling_any": "a slice endpoint, not an emitted equation: "
            + slice_endpoint_reason
            + "; its elements become the SMT variables ("
            + _q(_OBLIGATION, "flattened topological order, sans stelling_any")
            + ")",
        "stelling_assert": "a slice endpoint, not an emitted equation: "
            + slice_endpoint_reason,
    }
    for p in tr_not_em:
        w(f"- `{p}` — "
          f"{tr_not_em_reasons.get(p, 'no primitive-specific recorded reason')}")
    w("")

    # emission vs replay
    w("### Emission set vs replay surface")
    w("")
    em_not_rp = sorted(supported - replay)
    rp_not_em = sorted(replay - supported)
    if not em_not_rp and not rp_not_em:
        w(f"The emission set and the replay surface are equal "
          f"({len(supported)} primitives, both directions). The code records")
        w("this as an invariant: " + _q(_OBLIGATION,
          "Replay is what makes REFUTED self-certifying: a solver model is "
          "only ever promoted to a Witness after this module re-derives the "
          "violation in exact rational arithmetic, independently of the "
          "solver.") + " And: " + _q(_OBLIGATION,
          "Measured 2026-07-26: the two sets are currently EQUAL, in both "
          "directions. That equality is an invariant to preserve, not a "
          "coincidence to note") + ". It is asserted at import with the "
          "message " + _q(_OBLIGATION,
          "the exact-rational replay must cover exactly the emission set, "
          "or a witness can be produced that replay cannot independently "
          "confirm") + ".")
    else:
        w(f"Emitted but not replayed: {_prims(em_not_rp)}. Replayed but not "
          f"emitted: {_prims(rp_not_em)}.")
    w("")
    w("The replay path's scalar literal decoder is keyed by numpy dtype code")
    w(f"(`stelling.obligation._SCALAR_STRUCT_FMT`, "
      f"{len(_ob._SCALAR_STRUCT_FMT)} codes), not by primitive: "
      + _q(_OBLIGATION,
           "scalar decoders for size-1 ir.Array literals/consts "
           "(numpy dtype .str)") + ".")
    w("")

    # the two integer censuses
    w("### The two integer-semantics censuses")
    w("")
    shared = set(transfers) & supported
    both_computing = sorted(shared & computing & overflow)
    both_safe = sorted(shared & non_computing & int_safe)
    mismatched = sorted(
        (shared & computing & int_safe) | (shared & non_computing & overflow)
    )
    w(f"Of the {len(shared)} primitives in both a transfer registry and the")
    w(f"emission set, {len(both_computing)} are classified computing AND")
    w(f"overflow-guarded, {len(both_safe)} non-computing AND int-safe, and")
    if mismatched:
        w(f"{len(mismatched)} are classified differently by the two "
          f"censuses: {_prims(mismatched)}.")
    else:
        w("none is classified differently by the two censuses.")
    w("")
    w("Where both apply, the recorded relationship between the two guards")
    w("is: " + _q(_OBLIGATION,
      "SMT-LIB2 Reals are unbounded; jax integers wrap. Emitting a computed "
      "integer as a Real would let the solver prove a claim the program "
      "falsifies, so integer dtypes decline here — the emission is stricter "
      "than the transfer on purpose.") + " The transfer-side census's "
      "recorded charter is: " + _q(_PROPAGATE,
      "So the classification is mechanised instead of remembered. Every "
      "registered transfer is either COMPUTING — it can produce a numeric "
      "value its operands did not contain, so it carries the "
      "overflow-reachability guard — or NON-COMPUTING, with the reason "
      "recorded here.") + " The emission-side census's totality rule is: "
      + _q(_OBLIGATION,
      "Every emittable primitive is classified, and the union must be total "
      "over `_SUPPORTED`.") + ".")
    w("")
    tr_census_only = sorted((computing | non_computing) - supported)
    em_census_only = sorted((overflow | int_safe) - set(transfers))
    w(f"Censused on the transfer side only (not emitted, {len(tr_census_only)}): "
      f"{_prims(tr_census_only)}. Censused on the emission side only (no "
      f"transfer, {len(em_census_only)}): {_prims(em_census_only)}.")
    w("")

    # assume comparisons
    w("### Assume-narrowing comparisons vs the comparison set")
    w("")
    not_narrowing = sorted(compare - assume_cmps)
    extra = sorted(assume_cmps - compare)
    w(f"Of the {len(compare)} emitted comparisons, "
      f"{len(compare & assume_cmps)} can narrow through a constraining "
      f"assume; not narrowing: {_prims(not_narrowing)}"
      + (f"; narrowing but outside the comparison set: {_prims(extra)}"
         if extra else "") + ".")
    w("")
    if "ne" in not_narrowing:
        w("For `ne` the recorded reason is: " + _q(_PROPAGATE,
          "`ne` is a comparison but NOT here: excluding a single point from "
          "an interval does not narrow it (the hull of [lo, hi] \\ {k} is "
          "[lo, hi]) — it stays inert with the reason quoted.") + ".")
        others = [p for p in not_narrowing if p != "ne"]
        if others:
            w(f"For {_prims(others)}: no recorded reason.")
    elif not_narrowing:
        w(f"For {_prims(not_narrowing)}: no recorded reason.")
    w("")

    # -- appendices ----------------------------------------------------------
    w("## Recorded classification reasons")
    w("")
    w("The two reason registries, reproduced verbatim from the live dicts.")
    w("")
    w(f"### Emission int-safe reasons "
      f"(`stelling.obligation._INT_SAFE_EMITTED_REASONS`, "
      f"`{_OBLIGATION}:{_def_line(_OBLIGATION, '_INT_SAFE_EMITTED_REASONS')}`)")
    w("")
    out.extend(_table(
        ["primitive", "recorded reason"],
        [[f"`{p}`", _norm(r)]
         for p, r in sorted(_ob._INT_SAFE_EMITTED_REASONS.items())],
    ))
    w("")
    w(f"### Transfer non-computing exemptions "
      f"(`stelling.propagate._INT_NON_COMPUTING_EXEMPT`, "
      f"`{_PROPAGATE}:{_def_line(_PROPAGATE, '_INT_NON_COMPUTING_EXEMPT')}`)")
    w("")
    out.extend(_table(
        ["primitive", "recorded reason"],
        [[f"`{p}`", _norm(r)]
         for p, r in sorted(_pr._INT_NON_COMPUTING_EXEMPT.items())],
    ))
    w("")

    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    text = generate()
    if "--check" in argv:
        if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != text:
            print(f"{PAGE} differs from the live registries; regenerate "
                  f"with: python {_GENERATOR}", file=sys.stderr)
            return 1
        print(f"{PAGE} matches the live registries")
        return 0
    PAGE.write_text(text, encoding="utf-8")
    print(f"wrote {PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
