# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The routing manifest: what left `CHANGELOG.md`, and where it arrived.

TWO SECTIONS ARE ROUTED. `### Soundness fixes` was 2990 of `CHANGELOG.md`'s
3778 lines at `8f0adf2` — 79.1% of the file, against a released 0.1.0
section of 60 lines — and `### The eager construction-site detector (Mode
2), DEFAULT-OFF` was 242 of 1158 at `de80ad8`, 20.9%.
`DOCUMENTATION_ARCHITECTURE.md` §8.3 had already ruled that the Soundness
section carries one-liners and links to the ledger; it had drifted because
it routed detail to `evidence/soundness.yaml`, a file that was never built,
so the detail landed in the two files that exist. §8.3 now names
`SOUNDNESS.md` as the ledger. §8.3's letter governs the Soundness section
only; the Mode 2 section was routed on the ruling that its rationale
applies identically, and it is recorded here with its own `source_commit`,
its own span and its own blocks.

**Routing was chosen over summarising because it is checkable.** Every
block that left the changelog is pinned here by the sha256 of the text it
left as (`src_sha256`) and by the sha256 of the text it arrived as
(`dest_sha256`); `tests/test_soundness_routing.py` reads `SOUNDNESS.md` and
compares. For 68 of the 72 blocks the two hashes are equal, which is what
"verbatim" means here and is not a claim anyone has to take on trust.

**FOUR BLOCKS WERE EDITED IN TRANSIT AND ALL FOUR ARE DECLARED**, each
because the text it carried was wrong rather than because it read badly: an
unreproducible hash literal (`SF-0.2.0-59`), a solver workaround's obsolete
justification (`SF-0.2.0-51`), a route census stating 32 routes and 7
`unwatched` against a dict holding 33 and 8 (`SF-0.2.0-07`), and Mode 2's
own route census and its account of why the old fraction survived
(`M2-0.2.0-01`).

**AND THE MEASUREMENT THAT MAKES THAT SENTENCE MEAN ANYTHING WAS NOT TAKEN
UNTIL THE B8c FIXUP.** This paragraph used to end *"each edited block also
records `src_lines_not_carried` … and the test holds each to the number
recorded"*, and that was FALSE ABOUT THIS TREE'S OWN TEST: the only
assertion on the field was `<= 3`. Driven at `de80ad8`: `SF-0.2.0-59`'s
367-line body replaced by a three-line summary, `dest_sha256` updated as
any honest editor would, `src_lines_not_carried=3` declared, an `edit_note`
written — **`10 passed in 0.12s`**. Declaring `0` where two lines were
dropped passed too. An edit that is really a deletion is the thing this
manifest exists to prevent, and nothing prevented it.

It is measured now, twice over. Each edited block records
`src_lines_not_carried` AND `not_carried`, the lines themselves, and
`test_the_source_hashes_reproduce_from_git` recomputes both from
`source_commit:CHANGELOG.md` and asserts EQUALITY — not a bound. Quoting
the lines is what makes an edit reviewable: a summarisation with a note
attached would have to write every line it summarised away into this file,
where a reader will meet them. Two, three, three and eleven lines, in the
order the four are named above; the rest of all four blocks moved
untouched. Those numerals are restated here for a reader and are held to
nothing — each block's row carries its own, and the row is what the test
measures.

**Nothing was dropped.** `DROPPED` is empty and the test requires each
section's blocks to partition its span exactly, so a block that went
missing could not be silently absent from this file either — it would have
to be listed with a reason, and there is nothing to list.

**The limit, stated because it is real.** `src_sha256`, `src_span` and
`src_lines_not_carried` are all claims about a file this tree no longer
contains. They are verifiable — each section is re-derivable from
`git show <source_commit>:CHANGELOG.md` with the same splitter the routing
used — but only where git and that commit are present. In an sdist the
three git-gated tests skip, and what remains checkable there is the
destination: every hash, every anchor, and the two-way partition against
`CHANGELOG.md`. `tests/test_soundness_routing.py`'s module docstring says
which mutation class that leaves with NO guard at all, because it is not
the empty set.

The IDs are positional. They were minted once, at 0.2.0, in the order the
blocks stood in `CHANGELOG.md` at their section's `source_commit`, and
`src_span` is where each one stood.
"""

from __future__ import annotations

from typing import NamedTuple


class Block(NamedTuple):
    """One routed block. `kind` is `"entry"` (a soundness entry, which gets
    a `- **ID** — …` one-liner carrying a `Versions:` field) or `"context"`
    (a batch heading or an interstitial provenance note, which gets a
    heading line and no version field, because it makes no claim that has
    affected versions).

    `src_lines_not_carried` and `not_carried` are the same measurement
    twice: the number of the block's non-blank source lines that are not
    present verbatim in the destination, and those lines themselves.
    `test_the_source_hashes_reproduce_from_git` MEASURES both against the
    source and asserts equality with what is declared here — it does not
    take either on trust, and it does not merely bound them. Writing the
    lines out is what makes an edit reviewable: a "summarisation with a
    note attached" would have to quote every line it summarised away, in
    this file, where a reader will see them.
    """

    id: str
    anchor: str
    kind: str
    src_span: tuple[int, int]
    src_lines: int
    src_sha256: str
    dest_sha256: str
    src_lines_not_carried: int
    edit_note: str
    not_carried: tuple[str, ...] = ()


class Section(NamedTuple):
    """One routed `###` section of `CHANGELOG.md`.

    A routing is per-section, and every column here is per-section too:
    the commit the text left from, the span it occupied there, the `##`
    heading it arrived under in `SOUNDNESS.md`, and the ID prefix its
    blocks carry. `SOURCE_SPAN` was a bare module constant while there
    was one section, and a bare constant is a place a deletion can hide
    (shrink the span, drop the block it no longer covers, and every
    remaining check agrees) — `derive_source_span` recomputes it from the
    source file and holds this column to it.
    """

    key: str
    heading: str
    id_prefix: str
    source_commit: str
    source_span: tuple[int, int]
    dest_heading: str
    blocks: tuple[Block, ...]


#: Blocks deliberately NOT routed, each with the reason. Empty, and the
#: partition check is what makes that mean something.
DROPPED: tuple[tuple[str, str], ...] = ()

_SOUNDNESS_FIXES: tuple[Block, ...] = (
    Block(
        id="SF-0.2.0-01",
        anchor="sf-020-01",
        kind="entry",
        src_span=(577, 588),
        src_lines=12,
        src_sha256="00028fe3091170be393177a4a2d2dc6e0e522e02d90c95298b8ca38933af9c3a",
        dest_sha256="00028fe3091170be393177a4a2d2dc6e0e522e02d90c95298b8ca38933af9c3a",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-02",
        anchor="sf-020-02",
        kind="context",
        src_span=(590, 591),
        src_lines=2,
        src_sha256="ccd3594f828dc241e5003bf363cd4a06af46ff6784d33361b33ae77bfb2163ad",
        dest_sha256="ccd3594f828dc241e5003bf363cd4a06af46ff6784d33361b33ae77bfb2163ad",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-03",
        anchor="sf-020-03",
        kind="entry",
        src_span=(593, 637),
        src_lines=45,
        src_sha256="009f56f92bb12e7c1139ec9f724eceea7e426e699d1437bdb12bb7d4c429326d",
        dest_sha256="009f56f92bb12e7c1139ec9f724eceea7e426e699d1437bdb12bb7d4c429326d",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-04",
        anchor="sf-020-04",
        kind="entry",
        src_span=(639, 649),
        src_lines=11,
        src_sha256="3e871856a0d4c1ce827d70e2bd4c8427b11b4f784e859a881dfdd83fe75470c0",
        dest_sha256="3e871856a0d4c1ce827d70e2bd4c8427b11b4f784e859a881dfdd83fe75470c0",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-05",
        anchor="sf-020-05",
        kind="entry",
        src_span=(651, 663),
        src_lines=13,
        src_sha256="30efe46815e2412d5846ff13b522e3d7f9f0179d0d7191e07250c6883796bcc7",
        dest_sha256="30efe46815e2412d5846ff13b522e3d7f9f0179d0d7191e07250c6883796bcc7",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-06",
        anchor="sf-020-06",
        kind="entry",
        src_span=(665, 669),
        src_lines=5,
        src_sha256="a2c14d2cd242ee0797294ef010a6f68eb7cf141b220ea8dc0decfa9c4d01d18a",
        dest_sha256="a2c14d2cd242ee0797294ef010a6f68eb7cf141b220ea8dc0decfa9c4d01d18a",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-07",
        anchor="sf-020-07",
        kind="entry",
        src_span=(671, 691),
        src_lines=21,
        src_sha256="319d09cf527d0ae754196875b8d8dfca3923025a8608c1cafe026bde9242a455",
        dest_sha256="3d81c7c7c2ce0736e47497fc5037e16591633849a60401a472d5500a7adac00b",
        src_lines_not_carried=3,
        not_carried=(
            '  for each of 32 constant-construction routes — 17 `watched`, 7',
            '  `unwatched`, 3 `loud` (jax raises), 5 `deferred` (the constant reaches the',
            '  jaxpr and the convert transfer declines it) — and the suite MEASURES every',
        ),
        edit_note=(
            "the route census it states was wrong on arrival: 32 routes "
            "and 7 `unwatched` against a GATE_COVERAGE holding 33 and 8 "
            "since this batch's own fc98241; and the corrected 33/8 "
            "went stale in the B8c fixup, which enrolled two measured "
            "routes and moved it to 35/9. And the census line glossed "
            "the `deferred` bucket as the one the convert transfer "
            "declines, which is true of five of its six rows -- "
            "`jnp.take`'s `fill_value` has no `convert_element_type` in "
            "its jaxpr and is declined by an out-of-bounds `gather` -- so "
            "the gloss now claims only that something downstream declines "
            "the row, and the block names the two mechanisms and the "
            "per-row declaration that holds them. All three corrections "
            "stated in the block. No behaviour change."
        ),
    ),
    Block(
        id="SF-0.2.0-08",
        anchor="sf-020-08",
        kind="entry",
        src_span=(693, 716),
        src_lines=24,
        src_sha256="8ad4652601ba7e8c51c7b74f8785d9b4ad20f130dc7321ae616bbb24984c831b",
        dest_sha256="8ad4652601ba7e8c51c7b74f8785d9b4ad20f130dc7321ae616bbb24984c831b",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-09",
        anchor="sf-020-09",
        kind="entry",
        src_span=(718, 727),
        src_lines=10,
        src_sha256="9acd060f0c4087b23d1a8f7389fa2247d303e88817f086a1eea72562e9b77ffb",
        dest_sha256="9acd060f0c4087b23d1a8f7389fa2247d303e88817f086a1eea72562e9b77ffb",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-10",
        anchor="sf-020-10",
        kind="context",
        src_span=(729, 732),
        src_lines=4,
        src_sha256="2fb1e41580576340efbb84e9a2de92d64834e721b5b849ea9f87feded70e72bc",
        dest_sha256="2fb1e41580576340efbb84e9a2de92d64834e721b5b849ea9f87feded70e72bc",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-11",
        anchor="sf-020-11",
        kind="entry",
        src_span=(734, 750),
        src_lines=17,
        src_sha256="32552bdecf48a81fceb57a202029cc94c86f1f53ecfa427b701b187a50b66ef2",
        dest_sha256="32552bdecf48a81fceb57a202029cc94c86f1f53ecfa427b701b187a50b66ef2",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-12",
        anchor="sf-020-12",
        kind="entry",
        src_span=(752, 759),
        src_lines=8,
        src_sha256="11f961db58d9aa6d1e3bab644dd540e027699ce721fd0e706e65fb2936c6bf21",
        dest_sha256="11f961db58d9aa6d1e3bab644dd540e027699ce721fd0e706e65fb2936c6bf21",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-13",
        anchor="sf-020-13",
        kind="context",
        src_span=(761, 764),
        src_lines=4,
        src_sha256="21d6084afaf64d345c9a860bf9aeab639578968d00125b12c423d8adb64576d4",
        dest_sha256="21d6084afaf64d345c9a860bf9aeab639578968d00125b12c423d8adb64576d4",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-14",
        anchor="sf-020-14",
        kind="entry",
        src_span=(766, 1085),
        src_lines=320,
        src_sha256="e765342ea8269555cb49dd4fdec69ac0879c82e83433623fd7d70d6533a53248",
        dest_sha256="e765342ea8269555cb49dd4fdec69ac0879c82e83433623fd7d70d6533a53248",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-15",
        anchor="sf-020-15",
        kind="context",
        src_span=(1087, 1095),
        src_lines=9,
        src_sha256="2963e740d4304c76ec560d935c0302b82455468f8b507714157ba1f0ef8f4af5",
        dest_sha256="2963e740d4304c76ec560d935c0302b82455468f8b507714157ba1f0ef8f4af5",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-16",
        anchor="sf-020-16",
        kind="context",
        src_span=(1097, 1102),
        src_lines=6,
        src_sha256="ee040ab5d88f57d2b3e9965cbc0ce94b5f2458ea80310aecde25b062f069cb98",
        dest_sha256="ee040ab5d88f57d2b3e9965cbc0ce94b5f2458ea80310aecde25b062f069cb98",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-17",
        anchor="sf-020-17",
        kind="context",
        src_span=(1104, 1106),
        src_lines=3,
        src_sha256="407472ead10d6f88b4800652847cd61f5eaa5f5b5a408068eb161697321113b0",
        dest_sha256="407472ead10d6f88b4800652847cd61f5eaa5f5b5a408068eb161697321113b0",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-18",
        anchor="sf-020-18",
        kind="entry",
        src_span=(1108, 1150),
        src_lines=43,
        src_sha256="90d1d4bf570fbdc8b0d7d168a4d8c23c003dd3c0005d06bfa8c56d6ea398dacf",
        dest_sha256="90d1d4bf570fbdc8b0d7d168a4d8c23c003dd3c0005d06bfa8c56d6ea398dacf",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-19",
        anchor="sf-020-19",
        kind="entry",
        src_span=(1152, 1238),
        src_lines=87,
        src_sha256="f80fdd360d091d842d89cb128d4a1729960baf7b91c4e027f4231996d2075f51",
        dest_sha256="f80fdd360d091d842d89cb128d4a1729960baf7b91c4e027f4231996d2075f51",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-20",
        anchor="sf-020-20",
        kind="entry",
        src_span=(1240, 1254),
        src_lines=15,
        src_sha256="9cd26bd56d66404dd0194eda168386f6570f53dfc5da56be3d0dc4450c473328",
        dest_sha256="9cd26bd56d66404dd0194eda168386f6570f53dfc5da56be3d0dc4450c473328",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-21",
        anchor="sf-020-21",
        kind="entry",
        src_span=(1256, 1295),
        src_lines=40,
        src_sha256="cc9b42703288b9607e4273886b9116bb4df9077265136d1672e0ee3ee73c11db",
        dest_sha256="cc9b42703288b9607e4273886b9116bb4df9077265136d1672e0ee3ee73c11db",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-22",
        anchor="sf-020-22",
        kind="entry",
        src_span=(1297, 1310),
        src_lines=14,
        src_sha256="f13366337a7648a1255b0cf7be4875beb746cb1304df37cdacb275c676cb073d",
        dest_sha256="f13366337a7648a1255b0cf7be4875beb746cb1304df37cdacb275c676cb073d",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-23",
        anchor="sf-020-23",
        kind="entry",
        src_span=(1312, 1358),
        src_lines=47,
        src_sha256="0f0ac5bccccd7862783b25f7e193f798f2d6861b335657a40893e56f70f42a78",
        dest_sha256="0f0ac5bccccd7862783b25f7e193f798f2d6861b335657a40893e56f70f42a78",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-24",
        anchor="sf-020-24",
        kind="context",
        src_span=(1360, 1360),
        src_lines=1,
        src_sha256="82c5dfbfb447e8892d29b3be944fea667ea99af8c3daacc3f54f7bc7036f4fd6",
        dest_sha256="82c5dfbfb447e8892d29b3be944fea667ea99af8c3daacc3f54f7bc7036f4fd6",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-25",
        anchor="sf-020-25",
        kind="entry",
        src_span=(1362, 1462),
        src_lines=101,
        src_sha256="47767e74c19bac0e6e86d750d08d2276639e1d47886981923d925d5d0b2e67b7",
        dest_sha256="47767e74c19bac0e6e86d750d08d2276639e1d47886981923d925d5d0b2e67b7",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-26",
        anchor="sf-020-26",
        kind="context",
        src_span=(1464, 1466),
        src_lines=3,
        src_sha256="d813205c9d5d60c79bdbc01e9e2369876159af98f8b83cdd062a38801b69552f",
        dest_sha256="d813205c9d5d60c79bdbc01e9e2369876159af98f8b83cdd062a38801b69552f",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-27",
        anchor="sf-020-27",
        kind="entry",
        src_span=(1468, 1482),
        src_lines=15,
        src_sha256="d44e3f84660cdc8201d80c2acf57f8173d21e52a092ca27a18a59494195d2f04",
        dest_sha256="d44e3f84660cdc8201d80c2acf57f8173d21e52a092ca27a18a59494195d2f04",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-28",
        anchor="sf-020-28",
        kind="entry",
        src_span=(1484, 1500),
        src_lines=17,
        src_sha256="3e2da115ec25e3ffe4e44e78106b9f6e46855d6b0a0f6c7358c36f0691513329",
        dest_sha256="3e2da115ec25e3ffe4e44e78106b9f6e46855d6b0a0f6c7358c36f0691513329",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-29",
        anchor="sf-020-29",
        kind="entry",
        src_span=(1502, 1525),
        src_lines=24,
        src_sha256="80615ffa9e5e394710cc48173160ebe60ccef28f490ef1599e01412e8648b55f",
        dest_sha256="80615ffa9e5e394710cc48173160ebe60ccef28f490ef1599e01412e8648b55f",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-30",
        anchor="sf-020-30",
        kind="entry",
        src_span=(1527, 1541),
        src_lines=15,
        src_sha256="5d32e3c200e2816d3eb4c53ca1fcbc17759c74ce448d5b817255cf2d5c5c8570",
        dest_sha256="5d32e3c200e2816d3eb4c53ca1fcbc17759c74ce448d5b817255cf2d5c5c8570",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-31",
        anchor="sf-020-31",
        kind="entry",
        src_span=(1543, 1569),
        src_lines=27,
        src_sha256="1f1b32995ad48d7a74a19d83003e9a7d61f4195bd98a854e5d6791613c3e4882",
        dest_sha256="1f1b32995ad48d7a74a19d83003e9a7d61f4195bd98a854e5d6791613c3e4882",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-32",
        anchor="sf-020-32",
        kind="entry",
        src_span=(1571, 1610),
        src_lines=40,
        src_sha256="1e295d350452ee379d8c794ad6becbc144f514c229f7b0a66ff7fd8d26622902",
        dest_sha256="1e295d350452ee379d8c794ad6becbc144f514c229f7b0a66ff7fd8d26622902",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-33",
        anchor="sf-020-33",
        kind="entry",
        src_span=(1612, 1672),
        src_lines=61,
        src_sha256="ef1b9d61471d679ba54ae08313108a3c6c23ce23eb7c764866f7ad2cbce7aa70",
        dest_sha256="ef1b9d61471d679ba54ae08313108a3c6c23ce23eb7c764866f7ad2cbce7aa70",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-34",
        anchor="sf-020-34",
        kind="entry",
        src_span=(1674, 1747),
        src_lines=74,
        src_sha256="140133820b09c2a59c0245b559792bb7244742b941bcaafd821c64deda77b940",
        dest_sha256="140133820b09c2a59c0245b559792bb7244742b941bcaafd821c64deda77b940",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-35",
        anchor="sf-020-35",
        kind="entry",
        src_span=(1749, 1784),
        src_lines=36,
        src_sha256="2df819304fe7a283848458a11632d9c291d5ea2aa1c4449df2b3131881b151de",
        dest_sha256="2df819304fe7a283848458a11632d9c291d5ea2aa1c4449df2b3131881b151de",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-36",
        anchor="sf-020-36",
        kind="entry",
        src_span=(1786, 1803),
        src_lines=18,
        src_sha256="31ead6d2d23e99cad8877dd32366b5cac3672b12c9761368b843960ed626bb07",
        dest_sha256="31ead6d2d23e99cad8877dd32366b5cac3672b12c9761368b843960ed626bb07",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-37",
        anchor="sf-020-37",
        kind="entry",
        src_span=(1805, 1817),
        src_lines=13,
        src_sha256="60903120a9c5cda035707487e4cb52a1dd34e7c586e86c79424dc13f81541c88",
        dest_sha256="60903120a9c5cda035707487e4cb52a1dd34e7c586e86c79424dc13f81541c88",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-38",
        anchor="sf-020-38",
        kind="entry",
        src_span=(1819, 1857),
        src_lines=39,
        src_sha256="62ceb14219f2537dad2ddb725afe9942e9719fb6a7e389bc89dc8f7b77ca7d34",
        dest_sha256="62ceb14219f2537dad2ddb725afe9942e9719fb6a7e389bc89dc8f7b77ca7d34",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-39",
        anchor="sf-020-39",
        kind="entry",
        src_span=(1859, 1864),
        src_lines=6,
        src_sha256="489c3ad18a9c7a87bde68a30c3f2c6438a1875846c2f9b8b92f44bd8ef11f124",
        dest_sha256="489c3ad18a9c7a87bde68a30c3f2c6438a1875846c2f9b8b92f44bd8ef11f124",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-40",
        anchor="sf-020-40",
        kind="entry",
        src_span=(1866, 1869),
        src_lines=4,
        src_sha256="3217957b1eb997d3b59b38bb780d8c4d0c2b9e054ae7996e97efadf21527b58c",
        dest_sha256="3217957b1eb997d3b59b38bb780d8c4d0c2b9e054ae7996e97efadf21527b58c",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-41",
        anchor="sf-020-41",
        kind="entry",
        src_span=(1871, 1877),
        src_lines=7,
        src_sha256="7c7b2e8606ad3eeb602d98565886457a4c729dafc624a2d9fba465b08eca83ab",
        dest_sha256="7c7b2e8606ad3eeb602d98565886457a4c729dafc624a2d9fba465b08eca83ab",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-42",
        anchor="sf-020-42",
        kind="entry",
        src_span=(1879, 1882),
        src_lines=4,
        src_sha256="38c4a908cde504200b58574832139effc7813786806ecac376c1cc815daf385d",
        dest_sha256="38c4a908cde504200b58574832139effc7813786806ecac376c1cc815daf385d",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-43",
        anchor="sf-020-43",
        kind="entry",
        src_span=(1884, 1900),
        src_lines=17,
        src_sha256="4ed6067f0987b3b7cd71ba0d831441ba08ecb5ab9f3d027d15f8a0879674f8f2",
        dest_sha256="4ed6067f0987b3b7cd71ba0d831441ba08ecb5ab9f3d027d15f8a0879674f8f2",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-44",
        anchor="sf-020-44",
        kind="entry",
        src_span=(1902, 1919),
        src_lines=18,
        src_sha256="7b24333d3ab62058cd46a1b4bea47a74535f8415dac2420ae9d9d405856e5c9a",
        dest_sha256="7b24333d3ab62058cd46a1b4bea47a74535f8415dac2420ae9d9d405856e5c9a",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-45",
        anchor="sf-020-45",
        kind="entry",
        src_span=(1921, 1931),
        src_lines=11,
        src_sha256="b5a65b40c3acfadac02be1a1e2a0c85cdb12f0429b5a44e107aaec2299105f70",
        dest_sha256="b5a65b40c3acfadac02be1a1e2a0c85cdb12f0429b5a44e107aaec2299105f70",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-46",
        anchor="sf-020-46",
        kind="entry",
        src_span=(1933, 1965),
        src_lines=33,
        src_sha256="cae1bf46ac51e599f077a207732e0296a308a47a3b0a3c10862deaf19cba7616",
        dest_sha256="cae1bf46ac51e599f077a207732e0296a308a47a3b0a3c10862deaf19cba7616",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-47",
        anchor="sf-020-47",
        kind="entry",
        src_span=(1967, 1971),
        src_lines=5,
        src_sha256="f476624b627bf6497fee7bab59c40d16d3c4c737e73ffc89f887531dfb707b24",
        dest_sha256="f476624b627bf6497fee7bab59c40d16d3c4c737e73ffc89f887531dfb707b24",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-48",
        anchor="sf-020-48",
        kind="entry",
        src_span=(1973, 2025),
        src_lines=53,
        src_sha256="0e2db9204c710f690ef126ea75c857bff8745e63d21b5d5d2d58fcaf5ff7135a",
        dest_sha256="0e2db9204c710f690ef126ea75c857bff8745e63d21b5d5d2d58fcaf5ff7135a",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-49",
        anchor="sf-020-49",
        kind="entry",
        src_span=(2027, 2054),
        src_lines=28,
        src_sha256="46f59d0876ad8e2fe96b7d73ace700b2752d99c8012ceb6bca0ec776637081f7",
        dest_sha256="46f59d0876ad8e2fe96b7d73ace700b2752d99c8012ceb6bca0ec776637081f7",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-50",
        anchor="sf-020-50",
        kind="entry",
        src_span=(2056, 2136),
        src_lines=81,
        src_sha256="f8add2168f9578a46dc802ddfc4d1255f2db1d61d7939010517728676039f747",
        dest_sha256="f8add2168f9578a46dc802ddfc4d1255f2db1d61d7939010517728676039f747",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-51",
        anchor="sf-020-51",
        kind="entry",
        src_span=(2138, 2144),
        src_lines=7,
        src_sha256="6e9996eb5aa05ac9febf6fb7e2819c550e60c422cbd19174e9fb6a6f373ca6b0",
        dest_sha256="b39125907239bafce00dffc3f38950cf07ae562f3b3f4b2fe4b02f09259c3638",
        src_lines_not_carried=3,
        not_carried=(
            "  default `Solver()`. This restores the z3 cross-check on high-degree",
            "  polynomials (measured: d=80 from 10s+ timeout to 0.35-0.6s). The tactic",
            "  is activated automatically; cvc5 handles these natively.",
        ),
        edit_note="the stated reason (a degree-80 factoring pathology) names a case the emission cannot produce; replaced by the measured reason, with the re-derived z3 5.0.0 table. No behaviour change.",
    ),
    Block(
        id="SF-0.2.0-52",
        anchor="sf-020-52",
        kind="entry",
        src_span=(2146, 2151),
        src_lines=6,
        src_sha256="81b49a48ea36470760ae41a4222c1357b27b2b5e08eb59f69baf18d3b85f2d29",
        dest_sha256="81b49a48ea36470760ae41a4222c1357b27b2b5e08eb59f69baf18d3b85f2d29",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-53",
        anchor="sf-020-53",
        kind="entry",
        src_span=(2153, 2161),
        src_lines=9,
        src_sha256="dac576415c45eccc804f17e3f471593b089d4391860daad111dbe6b68b5d7a8b",
        dest_sha256="dac576415c45eccc804f17e3f471593b089d4391860daad111dbe6b68b5d7a8b",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-54",
        anchor="sf-020-54",
        kind="entry",
        src_span=(2163, 2165),
        src_lines=3,
        src_sha256="4bceb769b2a4bab22f8dc75fa04febf03be4544e17a70290f3d0217b20bdc698",
        dest_sha256="4bceb769b2a4bab22f8dc75fa04febf03be4544e17a70290f3d0217b20bdc698",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-55",
        anchor="sf-020-55",
        kind="entry",
        src_span=(2167, 2212),
        src_lines=46,
        src_sha256="6522dd9446a628fdabf4bde4ff624ca7df54dc20c74fd26b502d27575c41c20e",
        dest_sha256="6522dd9446a628fdabf4bde4ff624ca7df54dc20c74fd26b502d27575c41c20e",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-56",
        anchor="sf-020-56",
        kind="entry",
        src_span=(2214, 2492),
        src_lines=279,
        src_sha256="abb5c7fb570688396ae52df4e862e1a4e6506e12ba3d33cc86407a5994142a31",
        dest_sha256="abb5c7fb570688396ae52df4e862e1a4e6506e12ba3d33cc86407a5994142a31",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-57",
        anchor="sf-020-57",
        kind="entry",
        src_span=(2494, 2555),
        src_lines=62,
        src_sha256="b29383c3365ee04d2e6850f60ad110230d3439375f14ac7fa2585e71566e6aa3",
        dest_sha256="b29383c3365ee04d2e6850f60ad110230d3439375f14ac7fa2585e71566e6aa3",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-58",
        anchor="sf-020-58",
        kind="entry",
        src_span=(2557, 2594),
        src_lines=38,
        src_sha256="30e9d585509f902db83f735b4816542d46a8d4f459cf42e95216322e428481ce",
        dest_sha256="30e9d585509f902db83f735b4816542d46a8d4f459cf42e95216322e428481ce",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-59",
        anchor="sf-020-59",
        kind="entry",
        src_span=(2596, 2962),
        src_lines=367,
        src_sha256="d3581e50075343e79b60c8f20f4337f69e6bbae7ee912c42ccce895f040d3ef9",
        dest_sha256="d80f028b90596ef175aacb7e1b561b4d664305ac5a46432b83bb5cca2ac2bfb9",
        src_lines_not_carried=2,
        not_carried=(
            "  `dff95fc` and on `main` at `198a2b5` (both hashing to `64a0ce8d\u2026`) and",
            "  is a `TranscriptionError` here. The refusal is right \u2014",
        ),
        edit_note="replaces an unreproducible hash literal with the property it was standing for, re-derived across the two trees. No behaviour change.",
    ),
    Block(
        id="SF-0.2.0-60",
        anchor="sf-020-60",
        kind="entry",
        src_span=(2964, 3087),
        src_lines=124,
        src_sha256="05c29483b16d02ef6f4cd96ce780cf517e86d97d0bbeee5fce37bf072537a477",
        dest_sha256="05c29483b16d02ef6f4cd96ce780cf517e86d97d0bbeee5fce37bf072537a477",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-61",
        anchor="sf-020-61",
        kind="entry",
        src_span=(3089, 3229),
        src_lines=141,
        src_sha256="ecbe6fad3ff6e796577b7d7bc97169989afd5a98afb474be1cb3e383828d7200",
        dest_sha256="ecbe6fad3ff6e796577b7d7bc97169989afd5a98afb474be1cb3e383828d7200",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-62",
        anchor="sf-020-62",
        kind="entry",
        src_span=(3231, 3279),
        src_lines=49,
        src_sha256="2b354c8fb2660fa3bccbbbe5e4de552f8d883dc8f22a1c521321106e913a4790",
        dest_sha256="2b354c8fb2660fa3bccbbbe5e4de552f8d883dc8f22a1c521321106e913a4790",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-63",
        anchor="sf-020-63",
        kind="entry",
        src_span=(3281, 3403),
        src_lines=123,
        src_sha256="e67f611130231f1e8bb7ddd791dcc7c4c9df91d9ea8be75ae31ef4f8252d774b",
        dest_sha256="e67f611130231f1e8bb7ddd791dcc7c4c9df91d9ea8be75ae31ef4f8252d774b",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-64",
        anchor="sf-020-64",
        kind="entry",
        src_span=(3405, 3475),
        src_lines=71,
        src_sha256="aa2b6677f315d457bbff57c68112ab0960f158bb9c50464be9039260e598e81a",
        dest_sha256="aa2b6677f315d457bbff57c68112ab0960f158bb9c50464be9039260e598e81a",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-65",
        anchor="sf-020-65",
        kind="entry",
        src_span=(3477, 3484),
        src_lines=8,
        src_sha256="7b4a72aa9825cd0ebe16c0deebf47da9ca6aebbaa7358d260d048ed8e4498954",
        dest_sha256="7b4a72aa9825cd0ebe16c0deebf47da9ca6aebbaa7358d260d048ed8e4498954",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="SF-0.2.0-66",
        anchor="sf-020-66",
        kind="entry",
        src_span=(3486, 3563),
        src_lines=78,
        src_sha256="1f72b74efc77be87f826c2ebd77affabb68405ad3c8f599cf80ad31c70ceb0e5",
        dest_sha256="1f72b74efc77be87f826c2ebd77affabb68405ad3c8f599cf80ad31c70ceb0e5",
        src_lines_not_carried=0,
        edit_note="",
    ),
)

_MODE_2: tuple[Block, ...] = (
    Block(
        id="M2-0.2.0-01",
        anchor="m2-020-01",
        kind="entry",
        src_span=(134, 276),
        src_lines=143,
        src_sha256="b5f2a21a36aba075d22a2a8ee56fe2383d86fcf31bbc0ef9b8830025b4605287",
        dest_sha256="10251ade6a40f0f5c52fe5d22058c2d2c007e8071d67a054d251ef1173a6aac5",
        src_lines_not_carried=11,
        not_carried=(
            '  Six of the **eight** `unwatched` routes in',
            '  `lax.full`, `lax.full_like`, `lax.convert_element_type` and',
            '  `jnp.stack`-of-`full`, plus — measured, but rows of neither inventory —',
            "  `lax.select`-of-`full`, `jnp.take`'s `fill_value`, and a scoped",
            '  `with jax.disable_jit():`. Two numpy routes',
            '  remain and are named: `np.asarray(N).astype(dt)` is permanently unhookable',
            '  (`np.ndarray.astype` is an immutable type attribute) and',
            '  arithmetic gave it away (six closed plus two remaining is eight), and it',
            '  survived because the test asserted the NUMERATOR alone: `len(closed) == 6`',
            '  was true throughout. The denominator is asserted now. Measured at',
            '  `8f0adf2`: 33 routes — 17 `watched`, 8 `unwatched`, 3 `loud`, 5',
        ),
        edit_note=(
            'the route census it carried was corrected in the same commit '
            'that routed it: `lax.select`-of-`full` was enrolled in '
            'GATE_COVERAGE and EAGER_COVERAGE, which moves the closed '
            "fraction from six of eight to seven of nine; `jnp.take`'s "
            '`fill_value` was measured `deferred` rather than a hole the '
            'detector closes; and the published diagnosis of why the old '
            'fraction survived was replaced by the measured one. And the '
            'account of what declines that `deferred` row was corrected in '
            'turn, one commit later: it named the transfer that declines '
            'the other five `deferred` rows, of a route whose jaxpr holds '
            'no `convert_element_type` at all, and the mechanism is the '
            'definite out-of-bounds index on its `gather`. No behaviour '
            'change.'
        ),
    ),
    Block(
        id="M2-0.2.0-02",
        anchor="m2-020-02",
        kind="entry",
        src_span=(278, 319),
        src_lines=42,
        src_sha256="6f8dd6f4f4c4f7215d6478f282a25ee0d821c5d0e4c064f9f12105768d7e84ed",
        dest_sha256="6f8dd6f4f4c4f7215d6478f282a25ee0d821c5d0e4c064f9f12105768d7e84ed",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="M2-0.2.0-03",
        anchor="m2-020-03",
        kind="entry",
        src_span=(321, 338),
        src_lines=18,
        src_sha256="f48b48021d3647d550dc128aae456bcc45bc3fb70323a43a87df9d3c48c20060",
        dest_sha256="f48b48021d3647d550dc128aae456bcc45bc3fb70323a43a87df9d3c48c20060",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="M2-0.2.0-04",
        anchor="m2-020-04",
        kind="entry",
        src_span=(340, 347),
        src_lines=8,
        src_sha256="55a1e493d3b74d0cb28591c9adeffa782cdc59f41639f8def7843c2fd735a1fb",
        dest_sha256="55a1e493d3b74d0cb28591c9adeffa782cdc59f41639f8def7843c2fd735a1fb",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="M2-0.2.0-05",
        anchor="m2-020-05",
        kind="entry",
        src_span=(349, 352),
        src_lines=4,
        src_sha256="770f30a62de6063d5d799f80024035a7e9fd57d2764942d92c20f9b03c8d0afc",
        dest_sha256="770f30a62de6063d5d799f80024035a7e9fd57d2764942d92c20f9b03c8d0afc",
        src_lines_not_carried=0,
        edit_note="",
    ),
    Block(
        id="M2-0.2.0-06",
        anchor="m2-020-06",
        kind="entry",
        src_span=(354, 372),
        src_lines=19,
        src_sha256="3453e8daa832107ef08b42def1c43bd56e2bef79fd24811c4cf7a76c54a61fe2",
        dest_sha256="3453e8daa832107ef08b42def1c43bd56e2bef79fd24811c4cf7a76c54a61fe2",
        src_lines_not_carried=0,
        edit_note="",
    ),
)

SECTIONS: tuple[Section, ...] = (
    Section(
        key="soundness",
        heading="### Soundness fixes",
        id_prefix="SF-0.2.0-",
        source_commit="8f0adf2",
        source_span=(576, 3564),
        dest_heading=(
            "## 0.2.0 soundness-fix detail (routed from `CHANGELOG.md`)"
        ),
        blocks=_SOUNDNESS_FIXES,
    ),
    Section(
        key="mode2",
        heading="### The eager construction-site detector (Mode 2), DEFAULT-OFF",
        id_prefix="M2-0.2.0-",
        source_commit="de80ad8",
        source_span=(133, 373),
        dest_heading="## 0.2.0 Mode 2 detail (routed from `CHANGELOG.md`)",
        blocks=_MODE_2,
    ),
)

#: Every routed block, in section order then document order. The checks
#: that do not care which section a block came from read this.
ROUTED: tuple[Block, ...] = tuple(b for s in SECTIONS for b in s.blocks)
