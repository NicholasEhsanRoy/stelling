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
