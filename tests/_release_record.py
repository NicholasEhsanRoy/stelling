# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The release record is TWO files now, and a check that reads one is blind.

Batch B8c routed `CHANGELOG.md`'s `### Soundness fixes` detail into
`SOUNDNESS.md` — 2990 lines of predicate, measurement and derivation, block
for block, under `DOCUMENTATION_ARCHITECTURE.md` §8.3, which makes
`SOUNDNESS.md` the ledger and leaves the changelog one-liners that link to
it. `tests/test_soundness_routing.py` checks the move itself.

**FIVE test files went red on that move, in SEVEN functions, and every one
of them was RIGHT to.** They pin a sentence the project committed to — a
claim scoped, a route named, a figure re-derived — by reading
`CHANGELOG.md`. The sentence did not stop being committed to; it moved. So
the fix is not to delete the assertion, and it is not to weaken it: it is
to read the record where the record now is. The seven are named, so that
every sentence here is re-driveable by checking the named test out at
`8f0adf2` and running it against this tree:

* `test_assume_disclosure_claims.py::test_the_changelog_scopes_the_disclosure_claim_to_the_dispatch_path`
* `test_aval_lie_both_faces.py::test_the_element_count_census_covers_propagate_TOO`
* `test_canonicalization_routes.py::test_the_DISCLOSURES_name_exactly_these_routes`
* `test_ir_screen.py::test_the_batch_ships_an_attribution_table_that_adds_up`
* `test_ir_screen.py::test_an_attribution_row_may_not_quote_a_test_that_does_not_exist`
* `test_ir_screen.py::test_the_R1c_disclosure_EXHIBITS_its_pre_emption`
* `test_pow_row_gauge_jax.py::test_the_DOCSTRING_and_CHANGELOG_battery_SIZE_is_the_one_that_RAN`

**AND THE SIXTH FILE THE BATCH EDITED IS A DIFFERENT STORY, SEPARATED HERE
BECAUSE IT WAS TOLD AS PART OF THIS ONE.** `tests/test_tripwire_gate_coverage.py`
at `8f0adf2` runs **`11 passed`** against this tree. It reads no
documentation at all — `grep -cE 'read_text|CHANGELOG|SOUNDNESS|README|\\.md'`
over it returns **0** — so the routing could not have moved anything out
from under it. Its change in that batch (`len(unwatched) == 8` beside the
pre-existing `len(closed) == 6`) is an unrelated correction that rode
along, and calling it a sixth casualty of the routing filed a real fix
inside a story it has nothing to do with. The routing broke five files; the
denominator fix is its own entry.

**THE TRAP THIS CAMPAIGN KEEPS MEETING IS REAL, AND SMALLER THAN THE FIRST
ACCOUNT OF IT.** Three of the eight functions the batch touched pair *"the
retracted claim is absent"* with *"the scoped replacement is present"* over
the release record: `test_the_changelog_scopes_the_disclosure_claim_to_the_dispatch_path`'s
two legs over one `text`, `test_the_element_count_census_covers_propagate_TOO`'s
*"no caller anywhere"* against its named sites, and
`test_the_DISCLOSURES_name_exactly_these_routes`'s guarded-quotation scan
against its route list. The other five assert presence only. And **six of
the seven reds are on a presence leg**, which is the trap as advertised:
the absence leg passed because the paragraph was gone, so the retracted
wording was gone with it, and a checker built from absence alone would have
gone green on a routing that deleted these claims outright.

**The seventh red is the counterexample, and it points the other way.**
`test_the_DISCLOSURES_name_exactly_these_routes` failed on its **ABSENCE**
leg — *"SOUNDNESS.md still claims that only an `object.__setattr__` reaches
the residue"* — and its presence leg was never reached. The causal
direction is inverted too: nothing was deleted out of a page that check
read. The falsified sentence was quoted in `CHANGELOG.md`, which was not on
that check's page list, and the routing moved the quotation INTO
`SOUNDNESS.md`, which was. A page-list guard is only as wide as its list,
and what the routing did was widen what the list had to cover. That is a
third failure mode beside the two above, and it is the one a "read the
record where the record is" module cannot fix by itself.

`release_prose()` is deliberately a CONCATENATION and not a choice between
the files. A claim may live in either — a one-liner in the changelog, its
predicate in the ledger — and which one is an editorial decision that
should not red a test about the claim. What must never happen is the claim
being in NEITHER, and that is exactly what a concatenation still catches:
delete it from both and every caller goes red naming it.

**FOR A FIGURE, USE `release_records()`, AND THAT IS NOT A STYLE
PREFERENCE.** A concatenation widens a PRESENCE check and NARROWS a
`re.search` that reads a number out of the record, because `search` returns
the FIRST match: a correct digit in `CHANGELOG.md` masks a stale one in
`SOUNDNESS.md`, and the check passes while the ledger ships the wrong
number. Driven at this commit — the battery-size sentence written correctly
into `CHANGELOG.md` and as `999` into `SOUNDNESS.md` — the `release_prose()`
form ran **`1 passed`**. So a figure is checked per file, in every file of
the record that carries it, and `release_records()` is the accessor that
makes that possible.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The files that together are the 0.2.0 release record. Adding one widens
#: every presence check that reads through `release_prose`; it never
#: narrows one. It does not widen a FIGURE check — see `release_records`.
RELEASE_FILES: tuple[str, ...] = ("CHANGELOG.md", "SOUNDNESS.md")

#: THE VERSION-FIELD VOCABULARY, written once and read by both checks that
#: use it: `tests/test_soundness_routing.py` over `CHANGELOG.md`'s
#: one-liners, and `tests/test_soundness_log_reach.py` over `SOUNDNESS.md`'s
#: `## Log` bullets, which carry the same sentence in italics. Two closed
#: sets that agreed by hand would be two things to keep in step; this is
#: one thing.
#:
#: THREE PHRASES AND NOT TWO. The Log has to be able to say that a defect
#: reached the 0.1.0 pre-release builds and no release — 33 of its 54
#: bullets are that, and it was 39 until six bullets dated after the
#: `v0.1.0` tag were re-scoped on 2026-08-24 — and neither of the two the
#: changelog uses can say it.
#: `test_soundness_log_reach.py` derives the reached-release count from
#: these fields, and a count derived from a vocabulary that cannot express
#: a third of its entries is a count over the entries that happened to fit.
VERSION_FIELDS: tuple[str, ...] = (
    "Versions: 0.1.0 pre-release builds only.",
    "Versions: 0.2.0 development builds only.",
    "Versions: `v0.1.0` and 0.2.0 development builds.",
)

#: The subset a `CHANGELOG.md` 0.2.0 one-liner may carry. The first phrase
#: is excluded on purpose rather than by omission: an entry in the 0.2.0
#: changelog records a fix that landed in 0.2.0, so the defect was in 0.2.0
#: development builds by construction and "0.1.0 pre-release only" cannot
#: be true of it. A permitted phrase that no entry can use is an untested
#: branch of a rule, and this campaign has withdrawn several.
CHANGELOG_VERSION_FIELDS: tuple[str, ...] = VERSION_FIELDS[1:]


def release_records() -> tuple[tuple[str, str], ...]:
    """`((name, text), …)` — the release record, file by file.

    For a check that reads a FIGURE out of the record. A figure published
    anywhere in the record has to be right everywhere it is published, and
    that is a per-file question: over the concatenation one correct copy
    hides every stale copy behind it.
    """
    out = []
    for name in RELEASE_FILES:
        path = REPO / name
        assert path.is_file(), (
            f"{name} is not in this tree. The release record is "
            f"{list(RELEASE_FILES)} and a missing member would make every "
            f"presence check that reads through here quietly weaker."
        )
        out.append((name, path.read_text(encoding="utf-8")))
    return tuple(out)


def release_prose() -> str:
    """`CHANGELOG.md` and `SOUNDNESS.md`, concatenated, as text.

    Joined with a blank line so no claim can be formed by one file's last
    line running into the next file's first.

    For PRESENCE and for ABSENCE. NOT for a figure read with `re.search`,
    which sees the first match only — `release_records()` is that.
    """
    return "\n\n".join(text for _name, text in release_records())
