# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The release record is TWO files now, and a check that reads one is blind.

Batch B8c routed `CHANGELOG.md`'s `### Soundness fixes` detail into
`SOUNDNESS.md` — 2990 lines of predicate, measurement and derivation, block
for block, under `DOCUMENTATION_ARCHITECTURE.md` §8.3, which makes
`SOUNDNESS.md` the ledger and leaves the changelog one-liners that link to
it. `tests/test_soundness_routing.py` checks the move itself.

**Six tests went red on that move and every one of them was RIGHT to.**
They pin a sentence the project committed to — a claim scoped, a route
named, a figure re-derived — by reading `CHANGELOG.md`. The sentence did
not stop being committed to; it moved. So the fix is not to delete the
assertion, and it is not to weaken it: it is to read the record where the
record now is.

**AND THIS IS THE TRAP THIS CAMPAIGN KEEPS MEETING.** Each of those tests
has two legs — *"the retracted claim is absent"* and *"the scoped
replacement is present"*. The ABSENCE leg passed throughout the move, for
the worst possible reason: the paragraph was gone, so the retracted wording
was gone with it. Only the PRESENCE leg noticed anything. A checker built
from absence alone would have gone green on a routing that had deleted
every one of these claims outright, and this module exists so the presence
leg has somewhere true to look.

`release_prose()` is deliberately a CONCATENATION and not a choice between
the files. A claim may live in either — a one-liner in the changelog, its
predicate in the ledger — and which one is an editorial decision that
should not red a test about the claim. What must never happen is the claim
being in NEITHER, and that is exactly what a concatenation still catches:
delete it from both and every caller goes red naming it.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The files that together are the 0.2.0 release record. Adding one widens
#: every presence check that reads through here; it never narrows one.
RELEASE_FILES: tuple[str, ...] = ("CHANGELOG.md", "SOUNDNESS.md")


def release_prose() -> str:
    """`CHANGELOG.md` and `SOUNDNESS.md`, concatenated, as text.

    Joined with a blank line so no claim can be formed by one file's last
    line running into the next file's first.
    """
    parts = []
    for name in RELEASE_FILES:
        path = REPO / name
        assert path.is_file(), (
            f"{name} is not in this tree. The release record is "
            f"{list(RELEASE_FILES)} and a missing member would make every "
            f"presence check that reads through here quietly weaker."
        )
        parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)
