# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A name a document presents as EXISTING, that `src/` does not have.

`DOCUMENTATION_ARCHITECTURE.md` said *"`trust_boundary` field"*, *"It lives in
`TransferMeta`"*, *"The verdict artifact carries `opaque_params`"* and
*"`arithmetic: real-with-margin`"*. `grep -r` over `src/` returned nothing for
any of the four. Two of them turned out to be labels rather than machinery and
one is a mode that does not exist; a reader had no way to tell an implemented
name from an intended one, and neither did any check in this repository.

**THOSE NINE SENTENCES HAVE BEEN REWRITTEN AND THIS GATE IS WHAT HELD THEM TO
IT** — see :func:`test_a_documented_name_presented_as_existing_is_in_src` for
what each became. **The rewrite used none of the three exclusions**, and that
was the instruction: an unchecked checkbox around a sentence nobody intends to
tick is this defect one layer up, and a gate whose own failure text names the
escapes is a gate whose escapes are the first thing an author reaches for.
Each sentence instead says what is true — which artifact carries what, or that
a name is designed and not built.

**AND THAT MADE A HOUSE STYLE EXPLICIT, WHICH IS WORTH MORE THAN THE NINE
EDITS.** Construction 3 below reads a backticked ``key: value`` pair as *"an
artifact carries this pair"*, because that is what the pair form means in this
repository's prose — so a field position that does NOT exist is written with
its field and its value apart: *"the `arithmetic` row's `real-with-margin`
position"*, never ``  `arithmetic: real-with-margin` ``. The pair form is
reserved for pairs a reader could find in a verdict. Nothing here enforces
that convention on its own — it is the reason construction 3 can be as blunt
as it is without being wrong.

**THE HARD PART IS THE FALSE-POSITIVE RATE, AND IT IS THE DESIGN.** A document
may legitimately name a planned thing — this one is a *planning document* that
says so in its own header — and a lint that cannot tell a plan from a claim
gets silenced within a week, at which point it is worse than nothing. So this
file does not lint prose. It lints **four constructions**, each of which
asserts, in the indicative, that a name is a thing the code HAS:

1. an appositive category — ``the `X` field``, ``an `X` parameter``, and the
   copula form ``  `X` is a dataclass``. The noun is what makes it an
   assertion: *the `X` field* says there is a field.
2. ``  `X` exists``.
3. a field-and-value pair as an artifact would carry it — ``  `arithmetic:
   real-with-margin` `` — where the KEY is snake_case. The VALUE is the name
   checked; the key is checked too, and both must be findable.
4. an existence-or-location verb with a backticked object — *lives in*,
   *carries*, *is stored in*, *is attached to*, *rides in*, *is recorded in*,
   *is generated from*. *"It lives in `TransferMeta`"* is a claim about where
   a thing is, which entails the thing.

**AND THREE EXCLUSIONS THAT ARE CONSTRUCTIONS AND NOT A LIST**, which is what
keeps the allowlist below to two entries:

* **an unchecked markdown checkbox** — ``- [ ] `TransferMeta` exists with
  `tier`, …`` is an item on an acceptance checklist, and an unchecked box is
  definitionally not-yet-true. A CHECKED box (``- [x]``) is a claim that the
  work is done and is NOT excluded, which is the asymmetry that matters:
  ticking the box is exactly when the sentence starts asserting.
* **a fenced code block** — a `python` fence showing a proposed dataclass is
  an illustration of a design, not a statement about the tree.
* **a dotted name** — ``  `jax.config.jax_debug_key_reuse` ``,
  ``  `jax_md.util.safe_mask` ``, ``  `DotAlgorithmPreset.F32_F32_F32` ``. A
  qualified reference names somebody else's namespace, and prose does not
  reliably say whose. **This is a real limit, not a technicality**: it takes
  ``  `TransferMeta.tier` `` out of scope too, and `TransferMeta` is caught
  only because §5.1 also writes *"It lives in `TransferMeta`"* undotted.

**WHAT THIS DELIBERATELY DOES NOT COVER, so no one reads a green as more than
it is.** Every other way a document can name a thing that is not there:
narrative mention (*"the tier is data, not prose"*), a name in a table cell
with no category noun, a name inside a heading, a §-reference to a section
that describes machinery, an argument list, a filename. A name is also
"present in `src/`" here if it appears **anywhere** in the source text —
including in a comment or a docstring — which is deliberately the weak
direction: this gate is for names that are *nowhere*, and a stricter
definition (defined as a symbol, exported, reachable) would start reporting
names that exist and are merely spelled differently, which is the noise that
kills a lint.

**THE CORPUS IS THE PROJECT'S OWN PROSE**, by rule: every tracked `.md` under
`docs/` and `design/`, plus the architecture pages at the root. `SOUNDNESS.md`
and `CHANGELOG.md` are out, and not by oversight — they are DATED EVENT
RECORDS, where *"on 2026-08-06 `make_solver_verdict` gained a check"* is true
of that date whatever the tree does later, and they carry their own gates
(`tests/test_soundness_routing.py`, `tests/test_soundness_log_reach.py`).
`scratchpad/` is out for the same reason and one more: it is raw working
evidence a page cites, not prose that asserts an API.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _repo_files import text_files  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The record files, excluded from the corpus with the reason above. Named
#: from `_release_record.RELEASE_FILES` rather than retyped, so widening the
#: release record widens this exclusion with it instead of leaving a new
#: dated ledger silently linted as if it were a specification.
from _release_record import RELEASE_FILES  # noqa: E402

#: Directories whose `.md` files are this project's own prose about itself.
_PROSE_DIRECTORIES = ("docs/", "design/")

#: Root-level pages that are prose about this project. A root `.md` that is
#: not one of these, not a record file and not a licence notice is a NEW page
#: nobody decided about, and `test_the_corpus_is_a_rule_and_not_a_stale_list`
#: refuses it rather than letting it go unlinted.
_ROOT_PAGES = (
    "ARCHITECTURE.md",
    "DOCUMENTATION_ARCHITECTURE.md",
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
)

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_VALUE = r"[A-Za-z_][A-Za-z0-9_-]*"

#: The category nouns. A name followed by one of these is being given a kind,
#: and giving a thing a kind asserts the thing.
_CATEGORY = (
    "field", "fields", "type", "mode", "parameter", "flag", "enum", "class",
    "dataclass", "attribute", "method", "function", "option", "argument",
    "setting", "constant",
)

#: The existence-or-location verbs of construction 4.
_LOCATION_VERB = (
    r"lives? in", r"lived in", r"carries", r"carried in", r"stored in",
    r"is stored in", r"attached to", r"is attached to", r"rides in",
    r"recorded in", r"is recorded in", r"generated from",
)

_NOUN = "(?:" + "|".join(_CATEGORY) + ")"

_CONSTRUCTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "category-noun",
        re.compile(rf"`({_IDENT})`\s+{_NOUN}\b"),
    ),
    (
        "copula-category",
        re.compile(rf"`({_IDENT})`\s+(?:is|are)\s+(?:an?|the)\s+(?:\w+\s+){{0,2}}{_NOUN}\b"),
    ),
    (
        "exists",
        re.compile(rf"`({_IDENT})`\s+exists\b"),
    ),
    (
        "location-verb",
        re.compile(r"\b(?:" + "|".join(_LOCATION_VERB) + rf")\s+`({_IDENT})`"),
    ),
)

#: Construction 3, kept apart because it yields TWO names from one match and
#: the key must be snake_case. `XlaRuntimeError: UNIMPLEMENTED` is an
#: exception and its message, not a field and its value, and the case rule is
#: what tells them apart — by construction, not by naming it in a list.
_FIELD_VALUE = re.compile(rf"`([a-z][a-z0-9_]*)\s*:\s*({_VALUE})`")

#: Values that are literals rather than names. A `true` absent from `src/`
#: would be a finding about Python, not about this project.
_LITERALS = frozenset({"true", "false", "none", "null", "nan", "inf", "yes", "no"})

_UNCHECKED_BOX = re.compile(r"^\s*[-*+]\s*\[ \]")
_FENCE = re.compile(r"^\s*(?:```|~~~)")


@dataclasses.dataclass(frozen=True)
class Permitted:
    """One name the constructions above reach and that is not a defect."""

    name: str
    why: str


#: THE ALLOWLIST. Two entries, each a name the constructions cannot tell apart
#: from a stelling name and that a reader can check in one grep.
#:
#: Held by `test_every_permitted_name_is_still_needed`, which fails on an
#: entry that no longer earns its place — the discipline
#: `tests/_release_record.py` uses for its permitted version phrases, where an
#: option nothing exercises is a red. An allowlist nobody prunes becomes the
#: defect it was added to describe.
PERMITTED = (
    Permitted(
        "DotAlgorithmPreset",
        "a `jax.lax` name, spelled bare in DOCUMENTATION_ARCHITECTURE.md §2.3 "
        "(`\"in the `DotAlgorithmPreset` enum\"`) where the surrounding "
        "paragraph is about jax's API and cites the version it was verified "
        "against. The dotted-name exclusion catches its sibling "
        "`DotAlgorithmPreset.F32_F32_F32` by construction and cannot reach "
        "this one. NOTE FOR THE READER, because permitting it hides half a "
        "finding: §4.2 tells an implementer to pin precision with "
        "`lax.DotAlgorithmPreset` rather than `precision=`, and `src/` does "
        "neither. That is an unimplemented RECOMMENDATION, which is a "
        "different defect from a nonexistent name, and it is not this gate's",
    ),
    Permitted(
        "funclog2",
        "a field of the reproducer in diffrax issue #207, quoted in "
        "`design/mwe-census.md`'s census of upstream minimal working examples "
        "(`\"body: `funclog2` field + stoichiometry matrix inline\"`). The "
        "census's whole subject is other projects' code, so a name in it is "
        "never a claim about this one — and no construction can see that, "
        "because the sentence is an ordinary appositive",
    ),
)


def _in_source() -> str:
    """Every byte of `src/`, concatenated.

    A substring test and not a symbol table, deliberately — see the module
    docstring. The weak direction is the safe one for a gate whose subject is
    names that are NOWHERE.
    """
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(REPO.glob("src/**/*.py"))
    )


def corpus() -> list[tuple[str, str]]:
    """`(relative path, text)` for the prose this gate reads.

    Uses `_repo_files.text_files`, so a checkout that grew a `.venv` with a
    copy of the docs in it is not linted — the environment-dependence class
    that put four of five consecutive reds on `main`.
    """
    out = []
    for rel, path in text_files():
        if not rel.endswith(".md"):
            continue
        if rel in RELEASE_FILES:
            continue
        if rel.startswith(_PROSE_DIRECTORIES) or rel in _ROOT_PAGES:
            out.append((rel, path.read_text(encoding="utf-8")))
    return out


def asserted_names(text: str) -> list[tuple[int, str, str, str]]:
    """`(line, construction, name, quotation)` for one document.

    Fenced blocks and unchecked checkboxes are dropped here, where the rule
    is, and not by each caller.
    """
    found: list[tuple[int, str, str, str]] = []
    fenced = False
    for number, line in enumerate(text.split("\n"), 1):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced or _UNCHECKED_BOX.match(line):
            continue
        for label, pattern in _CONSTRUCTIONS:
            for hit in pattern.finditer(line):
                name = hit.group(1)
                if "." in name:  # pragma: no cover - _IDENT excludes dots
                    continue
                found.append((number, label, name, hit.group(0).strip()))
        for hit in _FIELD_VALUE.finditer(line):
            key, value = hit.group(1), hit.group(2)
            for name in (key, value):
                if name.lower() in _LITERALS:
                    continue
                found.append((number, "field-value", name, hit.group(0)))
    return found


def missing_names(
    documents: list[tuple[str, str]], source: str
) -> list[tuple[str, int, str, str, str]]:
    """Every asserted name that `src/` does not have, allowlist applied."""
    permitted = {entry.name for entry in PERMITTED}
    out = []
    for rel, text in documents:
        for line, label, name, quotation in asserted_names(text):
            if name in permitted or name in source:
                continue
            out.append((rel, line, label, name, quotation))
    return sorted(out)


def test_a_documented_name_presented_as_existing_is_in_src():
    """The gate. **RED on `main` at `115d771` — nine sites, four names.**

    Four names, and what each sentence says now. None of the four was
    allowlisted: an allowlist entry for the defect a gate was written to catch
    is the gate arriving pre-silenced, and `PERMITTED` is for names the
    CONSTRUCTIONS cannot judge, never for instances waiting on a fix.

    * `trust_boundary` — *"`trust_boundary` field"* in §1.4's responsibility
      table and *"`trust_boundary: jaxpr`"* in §2.5. A stamp key the stamp
      does not have. §1.4's row now names where the traced ⇒ compiled gap
      *is* recorded and says that no verdict field records it, and §2.4's row
      marks the key as design; the paragraph under §1.4's table used to say
      both open gaps *"are verdict fields"*, and now says which one is and
      which one is not, and that the asymmetry is a defect.
    * `TransferMeta` — *"It lives in `TransferMeta`"*, §5.1 rule 3. The tier
      rides in the registry and reaches the stamp as `transfer_tiers`;
      §10.1's dataclass was never built and §9.1 already listed it as
      deferred, so rule 3 now says where the tier IS and keeps its point,
      which was never the container but the direction — code to doc, never
      back.
    * `opaque_params` — *"The verdict artifact carries `opaque_params`"*,
      §2.6, and again in `design/transparent-primitives.md`. The type
      `OpaqueParam` IS in `src/stelling/ir.py`; the artifact key is not,
      which is exactly the case a coarser check would call satisfied. §2.6's
      three consequences are now written as the obligations they are, over a
      sentence that says nothing counts opaque params into a verdict, and the
      schema sketch says which of its keys are design.
    * `real-with-margin` — *"`arithmetic: real-with-margin`"*, §2.4, §2.5,
      §2.6 and §6.4. A position of a dial, and neither the position nor the
      dial is spelled that way in the tree: what records the arithmetic a
      verdict is judged in is `semantics`, and what records how endpoints are
      computed is `arithmetic_mode`. **No margin is computed anywhere**, so
      the sentences that leaned on one — §2.5's candidate row and §6.4's
      *"a claim about ℝ, with margin"* — say instead what does and does not
      absorb the difference.

    `SOUNDNESS.md` reached the same four names from the other end and is the
    ledger for them; this is the architecture document's own prose, and the
    two are meant to be read together rather than to repeat each other.
    """
    documents = corpus()
    assert documents, "the documentation corpus is empty"
    missing = missing_names(documents, _in_source())
    rendered = "\n  ".join(
        f"{rel}:{line}  [{label}]  {name!r} — {quotation}"
        for rel, line, label, name, quotation in missing
    )
    assert not missing, (
        f"{len(missing)} name(s) are presented as EXISTING in this project's "
        f"prose and appear nowhere in `src/`:\n  {rendered}\n\n"
        f"Each of these is one of four indicative constructions — a category "
        f"noun, `X exists`, a `key: value` pair, or an existence/location "
        f"verb — so the sentence says the code HAS the thing. Either the code "
        f"should, or the sentence should say it is planned: an unchecked "
        f"`- [ ]` checkbox and a fenced illustration are both outside this "
        f"gate by construction, and are how a document names a thing it "
        f"intends to build. A name that is genuinely somebody else's API goes "
        f"in `PERMITTED` with a reason."
    )


def test_the_four_constructions_are_each_driven():
    """Each construction, observed to fire, on planted prose.

    *A guard never observed to fire is not known to be a guard.* A
    construction that had gone inert — a renamed noun, a regex that stopped
    anchoring — would silently narrow this gate to the constructions that
    still work, and the gate would keep passing.
    """
    absent = "a_name_that_is_definitely_not_in_src_anywhere"
    plants = {
        "category-noun": f"The `{absent}` field is written into every stamp.",
        "copula-category": f"`{absent}` is a frozen dataclass.",
        "exists": f"`{absent}` exists on every verdict.",
        "location-verb": f"The tier lives in `{absent}` and is generated.",
        "field-value": f"A verdict carries `arithmetic: {absent}`.",
    }
    for label, sentence in plants.items():
        hits = [
            (found_label, name)
            for _line, found_label, name, _q in asserted_names(sentence)
        ]
        assert (label, absent) in hits, (
            f"the {label!r} construction did not fire on {sentence!r}; the "
            f"module saw {hits}. A construction that has gone inert narrows "
            f"this gate silently."
        )

    # THE CONTROL. Prose that names the same absent identifier without
    # asserting it — no category noun, no verb, no pair — must NOT fire, or
    # the gate is "a backtick anywhere" and will be switched off.
    for quiet in (
        f"See `{absent}` in the roadmap.",
        f"We considered `{absent}`.",
        f"`{absent}` would need a migration.",
    ):
        assert not asserted_names(quiet), (
            f"a bare mention fired this gate: {quiet!r}. A lint that reads "
            f"every backticked word is a lint everyone silences."
        )


def test_the_three_exclusions_are_each_driven():
    """The exclusions, driven — including the asymmetry that carries them.

    An exclusion nobody drives is a hole nobody measured. The checked-box
    half is the one that matters: if `- [x]` were excluded too, a document
    could assert anything by writing it as a completed checklist item, which
    is the shape acceptance checklists are written in.
    """
    absent = "a_name_that_is_definitely_not_in_src_anywhere"
    claim = f"`{absent}` exists on every verdict."

    assert asserted_names(claim), "the control sentence does not fire"
    assert not asserted_names(f"- [ ] {claim}"), (
        "an UNCHECKED checkbox fired this gate. An unticked acceptance item "
        "is not-yet-true by construction and is how a plan is written."
    )
    ticked = asserted_names(f"- [x] {claim}")
    assert ticked, (
        "a CHECKED checkbox did not fire this gate. Ticking the box is "
        "exactly the moment the sentence starts asserting, and excluding it "
        "would let a document claim anything in checklist form."
    )
    assert not asserted_names(f"```python\n{claim}\n```"), (
        "a fenced code block fired this gate. A fence showing a proposed "
        "dataclass illustrates a design; it does not describe the tree."
    )
    assert not asserted_names(f"`stelling.{absent}` exists on every verdict."), (
        "a DOTTED name fired this gate. A qualified reference names a "
        "namespace and prose does not reliably say whose."
    )


@pytest.mark.parametrize("entry", PERMITTED, ids=lambda e: e.name)
def test_every_permitted_name_is_still_needed(entry: Permitted):
    """No permitted-but-unnecessary entry. Both halves, both red-able.

    An entry earns its place only while BOTH are true: some construction in
    the corpus still reaches the name, and `src/` still does not have it. If
    the sentence is rewritten the first fails; if the name is implemented the
    second does. Either way the entry is dead and this says so, which is the
    difference between an allowlist and a list of things nobody looks at
    again.
    """
    assert entry.why.strip() and len(entry.why.split()) >= 12, (
        f"the entry for {entry.name!r} carries no reason a reader can check"
    )
    reached = [
        (rel, line, label, quotation)
        for rel, text in corpus()
        for line, label, name, quotation in asserted_names(text)
        if name == entry.name
    ]
    assert reached, (
        f"nothing in the documentation corpus presents {entry.name!r} as "
        f"existing any more, so permitting it excuses nothing. Delete the "
        f"entry: an allowlist nobody prunes becomes the defect."
    )
    assert entry.name not in _in_source(), (
        f"{entry.name!r} is in `src/` now, so it would pass this gate on its "
        f"own and the permission is dead weight. Delete the entry."
    )


def test_the_corpus_is_a_rule_and_not_a_stale_list():
    """Every root `.md` is either linted or excluded FOR A REASON.

    `_ROOT_PAGES` is the one enumerated thing in this file, and an
    enumeration of current members is how a gate goes quiet: a new root page
    would simply not be read, and nothing would say so. So the root is
    partitioned — linted, or a release record, and there is no third
    answer. A page added at the root fails here until somebody decides which
    it is.

    Sub-directories need no such check: `docs/` and `design/` are linted by
    PREFIX, so a file added under either is read the day it lands.
    """
    at_the_root = {
        rel for rel, _path in text_files()
        if rel.endswith(".md") and "/" not in rel
    }
    undecided = at_the_root - set(_ROOT_PAGES) - set(RELEASE_FILES)
    assert not undecided, (
        f"root markdown page(s) this gate neither lints nor excludes: "
        f"{sorted(undecided)}. Add it to `_ROOT_PAGES` if it is prose about "
        f"this project, or state why it is a dated record like "
        f"{list(RELEASE_FILES)}. A page nobody decided about is a page this "
        f"gate does not read and does not say it does not read."
    )
    # ... and anti-vacuity in the other direction: a name in `_ROOT_PAGES`
    # that is not in the checkout is a list that has outlived its pages, and
    # the partition above would be satisfied by an empty root.
    stale = set(_ROOT_PAGES) - at_the_root
    assert not stale and at_the_root, (
        f"`_ROOT_PAGES` names {sorted(stale)}, which the checkout does not "
        f"have. The list has outlived its pages."
    )
