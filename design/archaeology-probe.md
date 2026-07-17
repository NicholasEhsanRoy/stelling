# The archaeology probe — foresight or scar tissue, registered before reading

**Status:** REGISTRATION, 2026-07-18. Committed before any repository
history is read. The question: every kill so far measured an **end state**
(17 clamps, 11 `safe_*` symbols, six verbatim primals) — this probe asks
how the defences got there. Two stories fit the end state identically:
**foresight** (the author knew; the class is easy; a verifier adds nothing
to anyone) and **scar tissue** (the guard is the record of a bug that
already happened and cost real time). Opposite implications; the census
cannot separate them because it reads code, not history. Git can.

**Stated up front — this instrument can support and cannot falsify.**
A scar-tissue finding is positive evidence the class cost experienced
authors money in this repo. A foresight finding is ambiguous: *either* the
class is easy *or* the author paid for it somewhere git cannot see — and
the second still supports the population reading. Foresight must not be
read as a falsification, and this is a weaker instrument than the tracker
probe. A scar-tissue result is a fact *about* the kills: it reopens
nothing, licenses nothing beyond the byproduct policy
(`design/jax-verification-categories.md`), and is an input to the value
model only if it returns receipts.

## Precondition — checked first, excluding on failure

**History legibility.** For each repo: does the public history open with a
bulk import or squashed dump? A repo whose defences all read as
"first-commit" because the history *starts* at a dump returns **false
foresight** — the census's "a null at low coverage is not a null" lesson,
relocated. Repos with dump-shaped or squash-rewritten history are excluded
and reported as illegible, before any site is read.

## Corpus — fixed

| defence class | sites | method |
|---|---|---|
| jax-md `safe_mask` + the 11 `safe_*` symbols | definition sites in `jax_md` | `git log -S` per symbol; blame the definition; `--follow`/`-C` across renames |
| the 17 clamp sites (guard anatomy) | represented by their source-level guard expressions in the owning modules (jax-md `partition.py`/`smap`; diffrax's integrate/save path), located by grep and the interrogation's `source_info` | `git blame -C` the guard lines |
| the two lineax norm rules' Intentional sets | `lineax/_norm.py` (`_two_norm_jvp`, `_zero_grad_at_zero`) | `git log -L` the functions; `-S "Get zero gradient"` |
| the float-boundary machinery | `equinox/internal/_nextafter.py` (file + DAZ comment); diffrax's 11 `nextafter` call sites | file history; `git log -S nextafter` |

Repos: `jax-md/jax-md`, `patrick-kidger/diffrax`,
`patrick-kidger/equinox`, `patrick-kidger/lineax` (full clones).

## The discriminator — three signals per site, not one

1. **First-commit or later?** Present in the line's first version =
   foresight-shaped. Added to a line that already existed = scar-shaped:
   the code shipped without it.
2. **Does the introducing commit describe a failure?** `fix:`, "NaN
   when", a linked issue/PR.
3. **Does the linked issue exist and describe an incident?**

**Receipt** := added later + linked issue/commit message describing the
failure. Added-later with no story is **ambiguous-hardening**, its own
bucket, not a receipt.

## Bands — fixed

| finding | reading |
|---|---|
| ≥3 defence classes with receipts, across ≥2 libraries | **Scar tissue.** The kills are population facts: mature code has paid; new code has not. Licenses nothing beyond the byproduct policy |
| 1–2 receipts | **Weak.** Report per-class, never aggregated |
| 0 receipts, history legible | **Ambiguous, not falsified** — consistent with foresight *and* with pre-paid experience. Resist reading it as kill-confirmation |
| history illegible | **No result.** Say so |

## Prior, recorded before running (the proposer's, 0-for-3 with one survivor)

> `safe_mask` is scar tissue — an abstraction that named and centralised a
> pattern exists because someone got tired of writing it. The clamps are
> mixed.

## Secondary probe — same instrument, marked secondary

For the tracker probe's 20 hits: what shape were the fixes? {guard added,
detector added, logic fix, docs, won't-fix, user-side resolution} — read
from closing comments/PRs of the closed hits. This measures the frame's
cell 2 from the other side (are the alive classes' defences really
detective-only). Secondary: it does not eat the pass.

---

# Reading (2026-07-18 — after the registration commit)

## Precondition results

diffrax and equinox: **legible** (organic roots, 429/479 lines,
2021-07). jax-md: root is a 4,441-line OSS import (2019) — per-site rule
applied: sites present in the root are illegible, sites among the 1,003
open commits since are readable; **every site that mattered postdates the
root**. lineax: root is a 6,544-line dump (2023) — its norm rules are
present at the dump and **illegible there**; the trail crosses repos
("Moved norms into Lineax", 2023-12) back into diffrax, where the string
arrives in an infra commit (2022-11, "Upgraded to eqx.internal") —
ambiguous, cross-repo.

## The digs

| site | introduced | signals | bucket |
|---|---|---|---|
| jax-md `safe_mask` | `cea93ab` 2020-03-26, **not in root** | commit message: **"Fixed NaNs due to a JAX change in the behavior of np.sqrt at 0"** — the exact killed class, and the trigger was an upstream jax behaviour change | **RECEIPT** (NaN-at-zero) |
| jax-md `safe_sqrt`/`safe_norm`/`safe_acos`/`safe_atan2`… | 2022 (reaxff), 2025 (OPLS-AA), 2026 (UMA) — each with new feature work | first-commit-of-those-lines, no failure story | **foresight — downstream of the 2020 receipt**: the idiom, once minted in a fix, is applied proactively thereafter. The population-transfer mechanism, visible inside one repo |
| diffrax `nextafter` machinery | `8808098` 2021-08-01 (feature commit), then `0a8b1bf` 2021-08-17 **"Crash fixes"** touching the controller's nextafter lines (9 hits in the diff) | added-then-hardened, 16 days apart; message names the failure class; pre-release, no issue tracker yet | **RECEIPT, message-grade** (float-boundary) |
| diffrax `t1_clip_floor` | `4e8d745` 2025-08-02 **"Use 100 ULP's to clip timesteps close to t1 (#660)"** | PR-linked; and dfx#632's closing comment reads "Fixed in #660 and released in 0.7.1" — the full chain: 258-day incident → ULP guard added → release | **RECEIPT, full-grade** (float-boundary) |
| jax-md sentinel guard `where(dR < cutoff_sq, result, N)` | `8d97f31` 2026-06-09 **"cell list overflow bug fix at boundaries (#378)"** | PR-linked boundary fix — the guard-anatomy sentinel is itself the record of a boundary bug | **RECEIPT** (boundary completeness/indexing) |
| jax-md `PartitionError` detector | `511eb19` 2022-11-26 "Initial pass at neighbor list safety improvements" | post-root, intent-shaped message, no failure story | ambiguous-hardening |
| equinox `_nextafter.py` DAZ guard | not in the file's first version; `-S` first lands on a tooling commit (2023-03) | the failure narrative lives in the **code comment** ("JAX uses DAZ… check to fail near zero"), not the commit message | ambiguous-hardening (strict bucket), comment noted |
| lineax zero-grad norm rules | present at lineax's root dump; earlier diffrax introduction rides an infra commit | cross-repo, no failure story recoverable | ambiguous / illegible at origin |

## Band: **scar tissue — met**

**Three defence classes with receipts across two libraries**: NaN-at-zero
(jax-md, 2020), float-boundary (diffrax, 2021 message-grade and 2025
full-grade), boundary-completeness (jax-md, 2026, PR-linked). The
proposer's prior on `safe_mask` lands exactly. Per the registration this
is a fact *about* the kills — mature code has paid; the observed foresight
is measurably downstream of payments — and it licenses nothing beyond the
byproduct policy. No kill reopens.

## The arc nobody ordered

`4e8d745` fixed #632 by adding the ULP endpoint-clipping guard — and
**#756 is an infinite rejection loop caused by that guard's interaction
with adaptive step rejection.** The defence added for one incident
created the composition failure of the next: scar tissue generating scar
tissue at a seam, dated, in one repo. Recorded as evidence for the
interaction/composition row (L6) and for the frame — defences are
site-local; their interactions are global.

## Secondary probe (sampled, not exhaustive)

Fix shapes among closed hits: #632 → **guard added** (ULP clip, #660);
#378/#339 family → **logic fix + sentinel guard added**; #969 →
**detector-logic fix**; #756 → endpoint-handling fix in the same
machinery; #249 → same-day logic fix. Guards and detector repairs
dominate the sample — consistent with cell 2 (defences arrive after the
fact, and the preventable classes get preventive fixes while the global
classes get better detectors). Five sampled; marked as a sample.
