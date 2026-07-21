<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# ARCHITECTURE

Normative architecture rules. SOUNDNESS.md governs what a verdict may
claim; this file governs where things live and who may influence them.
History and evidence stay in `design/`; this file is the small set of
rules that bind future work.

## Rule 1 — Sockets live with the library they verify

A **socket** is a living verification surface for one library: its
harnesses and envelopes, its known answers, its fidelity gates and
mutation battery, its consequence declarations. Sockets encode
maintainer knowledge (admissible ranges, failure geometry, which guards
catch what) and must live where that knowledge lives: **the library's
repo, run by the library's CI.** stelling ships instruments; it does
not accumulate sockets.

A **campaign exhibit** is a frozen record of a completed evidence
campaign (the E2a exhibits, the blind lineax sweep, the probes).
Exhibits stay in `corpus/supply/`, pinned to the versions they measured,
edited only to keep them reproducible. The distinction is lifecycle:
exhibits are history, sockets are maintenance.

Corollary: stelling core never grows a library-shaped artifact.
Templates in core carry general mathematics only (a 2x2 conditioning
reduction is general; an unrolled assembly of one library's operator is
not).

## Rule 2 — Core prose is library-neutral; provenance is the one marked exception

Coupling lives in prose before it lives in code, and no ordinary test
guards prose. Two classes:

- **Outward references** — core text pointing at a specific library or
  at a library's socket as context ("the X attachment's shape",
  "library Y's operator") — **banned in `src/stelling/`**.
- **Provenance citations** — naming the real-world census contact that
  justified a transfer row or structural addition ("allowed-by-census
  structural addition from the <library> <trace> round"). These are the
  census discipline's memory and are **allowed**, marked: the line must
  carry `census` (or cite a `design/` record) on the same line.

Enforced structurally (a duty-enforced prose rule would not survive —
L18): a repo lint, run as both a pre-commit hook and a suite test,
fails on any banned library identifier in `src/stelling/` whose line
lacks a provenance marker. The banned list is explicit and extensible;
`tests/` is deliberately out of scope (census-binding tests name their
contacts as provenance by nature).

## Rule 3 — The trust split

- Anything able to influence a **VERIFIED** stays core-audited:
  transfer rows, emission, census entries grow only through the
  censused adversarial build process. Semantic extension from outside
  core is refused — unsupported primitives get honest UNKNOWN with the
  decline naming the primitive. (The one researched middle path,
  decomposition-with-proof, is a design idea, not a mechanism.)
- Library-side sockets may **pose questions** (harnesses through the
  standard funnels, provenance recorded in the stamp) and **declare
  consequences** (triage-classification input, never verdict input).
- **Fidelity instruments ship in stelling** (`stelling.fidelity`): a
  gate stack must not be inheritable without the means to gauge it —
  the structural form of L21. The gauging loop refuses to bless a
  stack with no mutations, a failing baseline, or an unexplained
  survivor.

## First instance — the MIME socket move (registered before execution)

The move: `corpus/supply/mime_lsq_conditioning.py` leaves stelling; the
socket, its MIME-specific mutation set (the L21 battery, currently
session-ephemeral — the decay clock that forces this move today), and a
README land in the MIME repo. `design/la-attachment.md` remains
stelling's record of the pass. Exhibits staying in corpus: the E2a
files, `mime_fvm.py`, `mime_fvm_regional.py`, `maddening_preconditions.py`,
`lineax_preconditions.py`, `la_contract_probe.py`, `tautology_test.py`,
`maddening_cfl.py`, `cf_run.py`, `exhibit_632.py`.

**Registered done-condition — the definition of "decoupled", fixed in
advance:**

1. With the MIME package **uninstalled** from the jax venv: full pytest
   suite green in both venvs;
2. `corpus/supply/la_contract_probe.py` exit 0 (stelling's own evidence
   needs no library installed);
3. the library-identifier lint passes over `src/stelling/`;
4. `import stelling, stelling.contracts, stelling.fidelity` clean
   without MIME present;
5. with MIME **reinstalled** (`--no-deps`; `jax==0.11.0` verified
   unmoved): the socket runs exit 0 from its MIME-side home with every
   KA status identical to `design/la-attachment.md`'s table;
6. no recorded verdict flips anywhere; suite counts recorded
   before/after.

Expected and allowed: the socket's query hashes may change (it gains
the shipped gauging loop and a new header); its KA **statuses** may
not.

## Next socket — MADDENING first

MIME is built on top of the MADDENING framework, so the second-framework
instance (L12) targets the base framework: a MADDENING socket built
as-if-library-side (fresh-context builder, public API and docs only, no
design/ context) whose friction log becomes the socket-kit requirements
list (L16). The MIME socket is later re-derived on that pattern rather
than the other way round.
