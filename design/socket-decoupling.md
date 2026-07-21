# The socket decoupling — reading against the registered condition

**Status:** PASS RECORD, 2026-07-22. The rule and the done-condition
were committed first (`ARCHITECTURE.md`, commit 173a555 — "the rule
precedes the move"); this file is the reading. The forcing event: the
L21 battery lived in a session scratchpad — the one artifact with a
decay clock.

## What shipped and what moved

- **`stelling.fidelity`** — the mutation-gauging loop, generalized and
  shipped (zero-dep, jax-free): `gauge(baseline, gates, mutations, *,
  residual)` with structural refusals in both directions — no battery,
  failing baseline, unexplained survivor, and *stale residual claim*
  (an "undetectable" mutation that a gate catches) are all loud
  errors. No adopter can now inherit a gate stack without inheriting
  the means to gauge it — the durable form of round-2's F1 warning.
  14 tests, jax-free, running in both venvs.
- **The lint** (ARCHITECTURE.md Rule 2) — `tests/test_prose_hygiene.py`
  plus a mirrored pre-commit hook: banned library identifiers in
  `src/stelling/` unless the line carries a `census`/`design/`
  provenance marker. The hook anchors its marker filter past the
  `path:lineno:` prefix so a marker in a file *path* (`census.py`)
  cannot mask an unmarked line.
- **De-MIME'd core prose** — the Scale paragraph keeps every measured
  number (11 terms/matrix, N = 46/47, 517/512) without the library
  name or corpus path; `coefficient_contrast` keeps `mu = 1 + chi`
  without the domain nouns; provenance comments normalized to carry
  their marker on the naming line. Comment/docstring edits only.
- **The socket moved** — `MIME/verification/stelling/`: the socket
  script (all KA hard-asserts intact), `battery.py` rebuilt on
  `stelling.fidelity.gauge` (8 gates × 11 mutations, the 5 algebraic
  survivors as the residual mapping, the two load-bearing facts
  hard-asserted: baseline passes everything; `reg_10x` caught by the
  starved M cross-check), and a README with version pins and the L21
  paragraph. `corpus/supply/mime_lsq_conditioning.py` deleted from
  stelling; the exhibits named in ARCHITECTURE.md stay.

## The reading (every item measured by the orchestrator, not only the builder)

| registered item | measured |
|---|---|
| 1. MIME uninstalled → suites green | venv-jax **968 passed** (bit-identical to with-MIME); venv-nojax **779 passed + 14 skipped** |
| 2. probe exit 0 without MIME | exit 0 |
| 3. lint passes over src/stelling/ | passed (test + hook) |
| 4. imports clean without MIME | `stelling, stelling.contracts, stelling.fidelity` clean |
| 5. reinstall (`--no-deps`), jax unmoved, socket from MIME home | jax 0.11.0; socket exit 0, KA statuses REFUTED / VERIFIED / UNKNOWN→VERIFIED / REFUTED — identical to `design/la-attachment.md`; battery exit 0 |
| 6. no status flips; counts recorded | none; 953→968 jax and 764→779 nojax = exactly the 15 new jax-free tests |

The suite counts being bit-for-bit identical with and without MIME
installed is the round-1 "tests must be library-free" discipline paying
out: decoupling required deleting one file and editing prose, nothing
structural.

MIME-side note: the socket landed on MIME's current branch
(`feat/navion-gradient-emns`, whose HEAD is the pinned 7ce1efb) — its
placement in MIME's branch flow is MIME's decision from here; the
socket is self-contained under `verification/stelling/`.

## What this unblocks

The scatter-add/segment_sum/stack censused row build (spec'd, fresh
blind builder) — new rows are gauged with the shipped
`stelling.fidelity` loop, the first internal customer of the
instrument. After the rows: sockets stop transcribing and start tracing
the real function, which shrinks the next socket (MADDENING first, per
ARCHITECTURE.md) to a harness plus known answers.
